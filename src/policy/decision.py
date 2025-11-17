"""
Notification decision making.
Orchestrates all components to make final send/no-send decisions.
"""
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import (
    CandidateNotification,
    ScoredNotification,
    Decision
)
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import TinyLLMClient
from src.policy.scoring import NotificationScorer

logger = logging.getLogger(__name__)


class NotificationDecisionMaker:
    """
    Makes final send/no-send decisions for notifications.

    Orchestrates:
    - Base models (click, complaint)
    - Fatigue model
    - LLM rating (for top candidates)
    - Scoring policy
    """

    def __init__(
        self,
        click_model: BaseClickModel,
        complaint_model: BaseComplaintModel,
        fatigue_model: FatigueModel,
        llm_client: Optional[TinyLLMClient] = None,
        scorer: Optional[NotificationScorer] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize decision maker.

        Args:
            click_model: Trained click prediction model
            complaint_model: Trained complaint prediction model
            fatigue_model: Fatigue computation model
            llm_client: Optional LLM client (uses fallback if None)
            scorer: Optional scorer (creates default if None)
            config: Configuration object
        """
        self.click_model = click_model
        self.complaint_model = complaint_model
        self.fatigue_model = fatigue_model
        self.llm_client = llm_client
        self.config = config if config is not None else DEFAULT_CONFIG

        if scorer is None:
            scorer = NotificationScorer(self.config)
        self.scorer = scorer

    def decide_single(
        self,
        candidate: CandidateNotification,
        use_llm: bool = True
    ) -> Decision:
        """
        Make decision for single notification candidate.

        Args:
            candidate: Notification candidate
            use_llm: Whether to use LLM for rating

        Returns:
            Decision object
        """
        # Get predictions from base models
        p_click = self._predict_click(candidate)
        p_complaint = self._predict_complaint(candidate)

        # Compute fatigue penalty
        fatigue_penalty = self.fatigue_model.compute_fatigue_penalty(
            candidate.user_features
        )

        # Get LLM rating if requested and available
        llm_rating = None
        if use_llm and self.llm_client is not None:
            llm_result = self.llm_client.rate_notification(candidate)
            llm_rating = llm_result['rating']

        # Score candidate
        scored = self.scorer.score_candidate(
            candidate,
            p_click,
            p_complaint,
            llm_rating,
            fatigue_penalty
        )

        # Make decision
        should_send = scored.final_score >= self.config.policy.score_threshold

        # Determine reason
        reason = self._generate_decision_reason(
            should_send,
            scored,
            fatigue_penalty
        )

        # Create decision
        decision = Decision(
            candidate=candidate,
            scored=scored,
            should_send=should_send,
            reason=reason,
            decision_timestamp=datetime.now()
        )

        return decision

    def decide_batch(
        self,
        candidates: List[CandidateNotification],
        use_llm_for_top_k: bool = True,
        top_k: Optional[int] = None
    ) -> List[Decision]:
        """
        Make decisions for batch of candidates.

        Efficient batch processing:
        1. Get base model predictions for all
        2. Compute fatigue penalties for all
        3. Score all candidates
        4. Use LLM only for top K candidates (if requested)
        5. Re-score top K with LLM ratings
        6. Make final decisions

        Args:
            candidates: List of candidates
            use_llm_for_top_k: Whether to use LLM for top candidates
            top_k: Number of top candidates to rate with LLM (uses config if None)

        Returns:
            List of Decision objects
        """
        if not candidates:
            return []

        if top_k is None:
            top_k = self.config.policy.top_k_for_llm

        # Step 1: Get base model predictions
        p_clicks = self._predict_click_batch(candidates)
        p_complaints = self._predict_complaint_batch(candidates)

        # Step 2: Compute fatigue penalties
        fatigue_penalties = [
            self.fatigue_model.compute_fatigue_penalty(c.user_features)
            for c in candidates
        ]

        # Step 3: Initial scoring (without LLM)
        scored_notifications = self.scorer.score_batch(
            candidates,
            p_clicks,
            p_complaints,
            llm_ratings=None,  # No LLM yet
            fatigue_penalties=fatigue_penalties
        )

        # Step 4: Optionally use LLM for top K
        llm_ratings = [None] * len(candidates)

        if use_llm_for_top_k and self.llm_client is not None and top_k > 0:
            # Get top K by initial score
            top_scored = self.scorer.get_top_k(scored_notifications, top_k)
            top_indices = [
                scored_notifications.index(s) for s in top_scored
            ]

            # Rate top K with LLM
            for idx in top_indices:
                try:
                    llm_result = self.llm_client.rate_notification(candidates[idx])
                    llm_ratings[idx] = llm_result['rating']
                except Exception as e:
                    logger.warning(f"LLM rating failed for candidate {idx}: {e}")
                    llm_ratings[idx] = self.config.llm.fallback_rating

            # Step 5: Re-score with LLM ratings
            scored_notifications = self.scorer.score_batch(
                candidates,
                p_clicks,
                p_complaints,
                llm_ratings=llm_ratings,
                fatigue_penalties=fatigue_penalties
            )

        # Step 6: Make decisions
        decisions = []
        for i, scored in enumerate(scored_notifications):
            should_send = scored.final_score >= self.config.policy.score_threshold

            reason = self._generate_decision_reason(
                should_send,
                scored,
                fatigue_penalties[i]
            )

            decision = Decision(
                candidate=candidates[i],
                scored=scored,
                should_send=should_send,
                reason=reason,
                decision_timestamp=datetime.now()
            )
            decisions.append(decision)

        return decisions

    def _predict_click(self, candidate: CandidateNotification) -> float:
        """Predict click probability for single candidate."""
        # Build feature dict from candidate
        features = self._extract_features(candidate)
        return self.click_model.predict_single(features)

    def _predict_complaint(self, candidate: CandidateNotification) -> float:
        """Predict complaint probability for single candidate."""
        features = self._extract_features(candidate)
        return self.complaint_model.predict_single(features)

    def _predict_click_batch(
        self,
        candidates: List[CandidateNotification]
    ) -> List[float]:
        """Predict click probabilities for batch."""
        import pandas as pd

        # Convert candidates to DataFrame
        df = self._candidates_to_dataframe(candidates)

        # Predict
        predictions = self.click_model.predict(df)

        return predictions.tolist()

    def _predict_complaint_batch(
        self,
        candidates: List[CandidateNotification]
    ) -> List[float]:
        """Predict complaint probabilities for batch."""
        import pandas as pd

        df = self._candidates_to_dataframe(candidates)
        predictions = self.complaint_model.predict(df)

        return predictions.tolist()

    def _extract_features(
        self,
        candidate: CandidateNotification
    ) -> Dict[str, float]:
        """
        Extract features from candidate for model prediction.

        Args:
            candidate: Notification candidate

        Returns:
            Dict of features
        """
        uf = candidate.user_features

        # Get channel and notification type encoding
        channel_map = {ch: idx for idx, ch in enumerate(self.config.features.channels)}
        type_map = {t: idx for idx, t in enumerate(self.config.features.notification_types)}

        channel_value = candidate.channel.value if hasattr(candidate.channel, 'value') else candidate.channel
        type_value = candidate.notification_type.value if hasattr(candidate.notification_type, 'value') else candidate.notification_type

        features = {
            # User features
            'notif_count_24h': uf.notif_count_24h,
            'notif_count_168h': uf.notif_count_7d,
            'notif_count_720h': uf.notif_count_30d,
            'click_count_168h': uf.click_count_7d,
            'click_count_720h': uf.click_count_30d,
            'open_rate_168h': uf.open_rate_7d,
            'open_rate_720h': uf.open_rate_30d,

            # Template features
            'channel_encoded': channel_map.get(channel_value, 0),
            'notif_type_encoded': type_map.get(type_value, 0),
            'template_hash': hash(candidate.template_id) % self.config.features.template_hash_dim,

            # Temporal features
            'hour_of_day': candidate.hour_of_day,
            'day_of_week': candidate.day_of_week,
            'is_weekend': 1 if candidate.day_of_week >= 5 else 0,
            'is_night': 1 if (candidate.hour_of_day < 6 or candidate.hour_of_day >= 22) else 0,

            # Recency features
            'hours_since_last_notif_168h': uf.hours_since_last_notification or 999,
            'hours_since_last_click_168h': uf.hours_since_last_click or 999,
        }

        return features

    def _candidates_to_dataframe(
        self,
        candidates: List[CandidateNotification]
    ):
        """Convert candidates to DataFrame for batch prediction."""
        import pandas as pd

        rows = []
        for candidate in candidates:
            features = self._extract_features(candidate)
            rows.append(features)

        return pd.DataFrame(rows)

    def _generate_decision_reason(
        self,
        should_send: bool,
        scored: ScoredNotification,
        fatigue_penalty: float
    ) -> str:
        """
        Generate human-readable reason for decision.

        Args:
            should_send: Whether to send
            scored: Scored notification
            fatigue_penalty: Fatigue penalty applied

        Returns:
            Reason string
        """
        if should_send:
            reason_parts = [
                f"Score {scored.final_score:.3f} above threshold {self.config.policy.score_threshold:.3f}",
                f"(click: {scored.p_click:.2%}, complaint: {scored.p_complaint:.2%}",
            ]

            if scored.llm_rating is not None:
                reason_parts.append(f"LLM: {scored.llm_rating}/5")

            if fatigue_penalty < 1.0:
                reason_parts.append(f"fatigue: {fatigue_penalty:.2f})")
            else:
                reason_parts.append(")")

            return " ".join(reason_parts)
        else:
            # Determine primary reason for rejection
            if fatigue_penalty < 0.3:
                return f"Rejected: High fatigue (penalty: {fatigue_penalty:.2f})"
            elif scored.p_complaint > 0.1:
                return f"Rejected: High complaint risk ({scored.p_complaint:.2%})"
            elif scored.base_score < 0:
                return f"Rejected: Negative base score ({scored.base_score:.3f})"
            else:
                return f"Rejected: Score {scored.final_score:.3f} below threshold {self.config.policy.score_threshold:.3f}"

    def get_decision_stats(
        self,
        decisions: List[Decision]
    ) -> Dict[str, any]:
        """
        Compute statistics over decisions.

        Args:
            decisions: List of decisions

        Returns:
            Dict with statistics
        """
        if not decisions:
            return {
                'total': 0,
                'sent': 0,
                'rejected': 0,
                'send_rate': 0.0,
                'mean_score': 0.0,
                'mean_score_sent': 0.0,
                'mean_score_rejected': 0.0
            }

        total = len(decisions)
        sent = sum(1 for d in decisions if d.should_send)
        rejected = total - sent

        all_scores = [d.scored.final_score for d in decisions]
        sent_scores = [d.scored.final_score for d in decisions if d.should_send]
        rejected_scores = [d.scored.final_score for d in decisions if not d.should_send]

        import numpy as np

        return {
            'total': total,
            'sent': sent,
            'rejected': rejected,
            'send_rate': sent / total if total > 0 else 0.0,
            'mean_score': float(np.mean(all_scores)) if all_scores else 0.0,
            'mean_score_sent': float(np.mean(sent_scores)) if sent_scores else 0.0,
            'mean_score_rejected': float(np.mean(rejected_scores)) if rejected_scores else 0.0
        }
