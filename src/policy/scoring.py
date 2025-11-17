"""
Notification scoring logic.
Combines base model predictions, LLM ratings, and fatigue penalties.
"""
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import CandidateNotification, ScoredNotification


class NotificationScorer:
    """
    Computes final scores for notification candidates.

    Combines:
    - Base model predictions (p_click, p_complaint)
    - LLM quality ratings
    - Fatigue penalties
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize scorer.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG

    def compute_base_score(
        self,
        p_click: float,
        p_complaint: float
    ) -> float:
        """
        Compute base score from click and complaint predictions.

        Formula: p_click - lambda * p_complaint

        Args:
            p_click: Predicted click probability
            p_complaint: Predicted complaint probability

        Returns:
            Base score
        """
        base_score = p_click - self.config.policy.lambda_comp * p_complaint
        return float(base_score)

    def normalize_llm_rating(self, rating: int) -> float:
        """
        Normalize LLM rating from 1-5 scale to 0-1 scale.

        Args:
            rating: LLM rating (1-5)

        Returns:
            Normalized rating (0-1)
        """
        # Convert 1-5 to 0-1: (rating - 1) / 4
        if rating < 1:
            rating = 1
        elif rating > 5:
            rating = 5

        normalized = (rating - 1) / 4.0
        return float(normalized)

    def compute_final_score(
        self,
        p_click: float,
        p_complaint: float,
        llm_rating: int,
        fatigue_penalty: float = 1.0
    ) -> float:
        """
        Compute final notification score.

        Formula:
        score = alpha * base_score + beta * llm_rating_normalized
        final_score = score * fatigue_penalty

        Args:
            p_click: Predicted click probability
            p_complaint: Predicted complaint probability
            llm_rating: LLM quality rating (1-5)
            fatigue_penalty: Fatigue penalty multiplier (0-1)

        Returns:
            Final score
        """
        # Compute base score
        base_score = self.compute_base_score(p_click, p_complaint)

        # Normalize LLM rating
        llm_normalized = self.normalize_llm_rating(llm_rating)

        # Combine scores
        combined_score = (
            self.config.policy.alpha * base_score +
            self.config.policy.beta * llm_normalized
        )

        # Apply fatigue penalty
        final_score = combined_score * fatigue_penalty

        return float(final_score)

    def score_candidate(
        self,
        candidate: CandidateNotification,
        p_click: float,
        p_complaint: float,
        llm_rating: Optional[int] = None,
        fatigue_penalty: float = 1.0
    ) -> ScoredNotification:
        """
        Create a ScoredNotification from predictions.

        Args:
            candidate: Notification candidate
            p_click: Predicted click probability
            p_complaint: Predicted complaint probability
            llm_rating: Optional LLM rating (uses fallback if None)
            fatigue_penalty: Fatigue penalty multiplier

        Returns:
            ScoredNotification object
        """
        # Use fallback rating if not provided
        if llm_rating is None:
            llm_rating = self.config.llm.fallback_rating

        # Compute base score
        base_score = self.compute_base_score(p_click, p_complaint)

        # Compute final score
        final_score = self.compute_final_score(
            p_click, p_complaint, llm_rating, fatigue_penalty
        )

        # Create scored notification
        scored = ScoredNotification(
            candidate=candidate,
            p_click=p_click,
            p_complaint=p_complaint,
            base_score=base_score,
            llm_rating=llm_rating,
            fatigue_penalty=fatigue_penalty,
            final_score=final_score
        )

        return scored

    def score_batch(
        self,
        candidates: List[CandidateNotification],
        p_clicks: List[float],
        p_complaints: List[float],
        llm_ratings: Optional[List[int]] = None,
        fatigue_penalties: Optional[List[float]] = None
    ) -> List[ScoredNotification]:
        """
        Score multiple candidates.

        Args:
            candidates: List of candidates
            p_clicks: List of click probabilities
            p_complaints: List of complaint probabilities
            llm_ratings: Optional list of LLM ratings
            fatigue_penalties: Optional list of fatigue penalties

        Returns:
            List of ScoredNotification objects
        """
        n = len(candidates)
        assert len(p_clicks) == n
        assert len(p_complaints) == n

        # Use defaults if not provided
        if llm_ratings is None:
            llm_ratings = [self.config.llm.fallback_rating] * n
        if fatigue_penalties is None:
            fatigue_penalties = [1.0] * n

        assert len(llm_ratings) == n
        assert len(fatigue_penalties) == n

        # Score each candidate
        scored_notifications = []
        for i in range(n):
            scored = self.score_candidate(
                candidates[i],
                p_clicks[i],
                p_complaints[i],
                llm_ratings[i],
                fatigue_penalties[i]
            )
            scored_notifications.append(scored)

        return scored_notifications

    def rank_candidates(
        self,
        scored_notifications: List[ScoredNotification]
    ) -> List[ScoredNotification]:
        """
        Rank scored notifications by final score (descending).

        Args:
            scored_notifications: List of scored notifications

        Returns:
            List sorted by final_score (highest first)
        """
        return sorted(
            scored_notifications,
            key=lambda x: x.final_score,
            reverse=True
        )

    def get_top_k(
        self,
        scored_notifications: List[ScoredNotification],
        k: int
    ) -> List[ScoredNotification]:
        """
        Get top K candidates by score.

        Args:
            scored_notifications: List of scored notifications
            k: Number of top candidates to return

        Returns:
            Top K scored notifications
        """
        ranked = self.rank_candidates(scored_notifications)
        return ranked[:k]

    def compute_score_statistics(
        self,
        scored_notifications: List[ScoredNotification]
    ) -> Dict[str, float]:
        """
        Compute statistics over scored notifications.

        Args:
            scored_notifications: List of scored notifications

        Returns:
            Dict with statistics
        """
        if not scored_notifications:
            return {
                'count': 0,
                'mean_score': 0.0,
                'median_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'std_score': 0.0
            }

        scores = [s.final_score for s in scored_notifications]

        return {
            'count': len(scores),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores))
        }


class AdaptiveScorer(NotificationScorer):
    """
    Adaptive scorer that adjusts weights based on model performance.

    Can tune alpha/beta balance based on historical accuracy.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize adaptive scorer."""
        super().__init__(config)
        self.alpha = self.config.policy.alpha
        self.beta = self.config.policy.beta

    def update_weights(
        self,
        base_model_accuracy: float,
        llm_accuracy: float
    ) -> None:
        """
        Update alpha/beta weights based on component accuracy.

        Args:
            base_model_accuracy: Historical accuracy of base models
            llm_accuracy: Historical accuracy of LLM ratings
        """
        # Normalize accuracies
        total = base_model_accuracy + llm_accuracy
        if total > 0:
            self.alpha = base_model_accuracy / total
            self.beta = llm_accuracy / total
        else:
            # Fallback to config defaults
            self.alpha = self.config.policy.alpha
            self.beta = self.config.policy.beta

        # Ensure they sum to ~1.0
        total_weight = self.alpha + self.beta
        if total_weight > 0:
            self.alpha = self.alpha / total_weight
            self.beta = self.beta / total_weight

    def compute_final_score(
        self,
        p_click: float,
        p_complaint: float,
        llm_rating: int,
        fatigue_penalty: float = 1.0
    ) -> float:
        """
        Compute final score using adaptive weights.

        Args:
            p_click: Predicted click probability
            p_complaint: Predicted complaint probability
            llm_rating: LLM quality rating (1-5)
            fatigue_penalty: Fatigue penalty multiplier (0-1)

        Returns:
            Final score
        """
        # Compute base score
        base_score = self.compute_base_score(p_click, p_complaint)

        # Normalize LLM rating
        llm_normalized = self.normalize_llm_rating(llm_rating)

        # Combine scores with adaptive weights
        combined_score = (
            self.alpha * base_score +
            self.beta * llm_normalized
        )

        # Apply fatigue penalty
        final_score = combined_score * fatigue_penalty

        return float(final_score)

    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current alpha/beta weights.

        Returns:
            Dict with weights
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta
        }
