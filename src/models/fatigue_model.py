"""
Notification fatigue modeling.
Adjusts scores based on recent notification volume and user engagement patterns.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from datetime import datetime, timedelta

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import UserFeatures


class FatigueModel:
    """
    Models user fatigue from notifications.

    Reduces send probability when user receives too many notifications
    in a short time period, especially if they're not engaging.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize fatigue model.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG

    def compute_fatigue_penalty(
        self,
        user_features: UserFeatures
    ) -> float:
        """
        Compute fatigue penalty multiplier (0.0 to 1.0).

        Lower values indicate higher fatigue (stronger penalty).
        Returns 1.0 for no penalty, lower values for fatigued users.

        Args:
            user_features: User feature object with notification history

        Returns:
            Fatigue multiplier in range [0.0, 1.0]
        """
        penalties = []

        # Penalty 1: Too many notifications in 24h
        penalty_24h = self._compute_volume_penalty(
            user_features.notif_count_24h,
            threshold=self.config.policy.max_notifications_per_day,
            max_count=self.config.policy.max_notifications_per_day * 2
        )
        penalties.append(penalty_24h)

        # Penalty 2: Too many notifications in 7d
        penalty_7d = self._compute_volume_penalty(
            user_features.notif_count_7d,
            threshold=self.config.policy.max_notifications_per_day * 7,
            max_count=self.config.policy.max_notifications_per_day * 14
        )
        penalties.append(penalty_7d)

        # Penalty 3: Low open rate indicates fatigue
        if user_features.notif_count_7d >= 5:  # Only if enough history
            penalty_engagement = self._compute_engagement_penalty(
                user_features.open_rate_7d,
                min_acceptable=0.05  # 5% minimum engagement
            )
            penalties.append(penalty_engagement)

        # Penalty 4: Recent notification sent too soon
        if user_features.hours_since_last_notification is not None:
            penalty_recency = self._compute_recency_penalty(
                user_features.hours_since_last_notification,
                min_gap_hours=1.0  # At least 1 hour between notifications
            )
            penalties.append(penalty_recency)

        # Combine penalties (multiplicative)
        final_penalty = np.prod(penalties)

        return float(final_penalty)

    def _compute_volume_penalty(
        self,
        count: int,
        threshold: int,
        max_count: int
    ) -> float:
        """
        Compute penalty for notification volume.

        Args:
            count: Number of notifications sent
            threshold: Count at which penalty begins
            max_count: Count at which penalty is maximum (0.0)

        Returns:
            Penalty multiplier in [0.0, 1.0]
        """
        if count <= threshold:
            return 1.0

        if count >= max_count:
            return 0.0

        # Linear decay between threshold and max_count
        penalty = 1.0 - (count - threshold) / (max_count - threshold)
        return max(0.0, min(1.0, penalty))

    def _compute_engagement_penalty(
        self,
        open_rate: float,
        min_acceptable: float
    ) -> float:
        """
        Compute penalty for low engagement.

        Args:
            open_rate: User's recent open rate
            min_acceptable: Minimum acceptable open rate

        Returns:
            Penalty multiplier in [0.0, 1.0]
        """
        if open_rate >= min_acceptable * 2:
            return 1.0

        if open_rate <= 0:
            return 0.3  # Don't completely block, but heavily penalize

        # Smooth penalty for low engagement
        ratio = open_rate / min_acceptable
        penalty = 0.3 + 0.7 * min(1.0, ratio)

        return penalty

    def _compute_recency_penalty(
        self,
        hours_since_last: float,
        min_gap_hours: float
    ) -> float:
        """
        Compute penalty for sending too soon after last notification.

        Args:
            hours_since_last: Hours since last notification
            min_gap_hours: Minimum desired gap in hours

        Returns:
            Penalty multiplier in [0.0, 1.0]
        """
        if hours_since_last >= min_gap_hours * 2:
            return 1.0

        if hours_since_last <= min_gap_hours * 0.5:
            return 0.1  # Strong penalty for very recent notifications

        # Smooth penalty
        ratio = hours_since_last / min_gap_hours
        penalty = min(1.0, ratio)

        return penalty

    def should_suppress(
        self,
        user_features: UserFeatures,
        threshold: float = 0.3
    ) -> bool:
        """
        Determine if notification should be suppressed due to fatigue.

        Args:
            user_features: User feature object
            threshold: Fatigue penalty below which to suppress

        Returns:
            True if should suppress, False otherwise
        """
        penalty = self.compute_fatigue_penalty(user_features)
        return penalty < threshold

    def adjust_score(
        self,
        base_score: float,
        user_features: UserFeatures
    ) -> float:
        """
        Adjust notification score for fatigue.

        Args:
            base_score: Base score from click/complaint models
            user_features: User feature object

        Returns:
            Adjusted score
        """
        fatigue_penalty = self.compute_fatigue_penalty(user_features)
        adjusted = base_score * fatigue_penalty

        return adjusted

    def get_fatigue_stats(
        self,
        user_features: UserFeatures
    ) -> Dict[str, float]:
        """
        Get detailed fatigue statistics for debugging.

        Args:
            user_features: User feature object

        Returns:
            Dictionary with fatigue components
        """
        stats = {}

        # Overall penalty
        stats['overall_penalty'] = self.compute_fatigue_penalty(user_features)

        # Component penalties
        stats['volume_24h_penalty'] = self._compute_volume_penalty(
            user_features.notif_count_24h,
            threshold=self.config.policy.max_notifications_per_day,
            max_count=self.config.policy.max_notifications_per_day * 2
        )

        stats['volume_7d_penalty'] = self._compute_volume_penalty(
            user_features.notif_count_7d,
            threshold=self.config.policy.max_notifications_per_day * 7,
            max_count=self.config.policy.max_notifications_per_day * 14
        )

        if user_features.notif_count_7d >= 5:
            stats['engagement_penalty'] = self._compute_engagement_penalty(
                user_features.open_rate_7d,
                min_acceptable=0.05
            )
        else:
            stats['engagement_penalty'] = 1.0

        if user_features.hours_since_last_notification is not None:
            stats['recency_penalty'] = self._compute_recency_penalty(
                user_features.hours_since_last_notification,
                min_gap_hours=1.0
            )
        else:
            stats['recency_penalty'] = 1.0

        # User state
        stats['notif_count_24h'] = user_features.notif_count_24h
        stats['notif_count_7d'] = user_features.notif_count_7d
        stats['open_rate_7d'] = user_features.open_rate_7d
        stats['hours_since_last'] = user_features.hours_since_last_notification or -1

        return stats


class AdaptiveFatigueModel(FatigueModel):
    """
    Adaptive fatigue model that adjusts thresholds based on user behavior.

    For power users who engage frequently, allows more notifications.
    For less engaged users, becomes more conservative.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize adaptive fatigue model."""
        super().__init__(config)

    def compute_fatigue_penalty(
        self,
        user_features: UserFeatures
    ) -> float:
        """
        Compute fatigue penalty with adaptive thresholds.

        Args:
            user_features: User feature object

        Returns:
            Fatigue multiplier in range [0.0, 1.0]
        """
        # Determine user engagement tier
        engagement_tier = self._classify_engagement(user_features)

        # Adjust thresholds based on tier
        if engagement_tier == 'high':
            daily_threshold = int(self.config.policy.max_notifications_per_day * 1.5)
            min_gap_hours = 0.5
        elif engagement_tier == 'medium':
            daily_threshold = self.config.policy.max_notifications_per_day
            min_gap_hours = 1.0
        else:  # low
            daily_threshold = int(self.config.policy.max_notifications_per_day * 0.5)
            min_gap_hours = 2.0

        # Compute penalties with adjusted thresholds
        penalties = []

        penalty_24h = self._compute_volume_penalty(
            user_features.notif_count_24h,
            threshold=daily_threshold,
            max_count=daily_threshold * 2
        )
        penalties.append(penalty_24h)

        penalty_7d = self._compute_volume_penalty(
            user_features.notif_count_7d,
            threshold=daily_threshold * 7,
            max_count=daily_threshold * 14
        )
        penalties.append(penalty_7d)

        if user_features.notif_count_7d >= 5:
            penalty_engagement = self._compute_engagement_penalty(
                user_features.open_rate_7d,
                min_acceptable=0.05
            )
            penalties.append(penalty_engagement)

        if user_features.hours_since_last_notification is not None:
            penalty_recency = self._compute_recency_penalty(
                user_features.hours_since_last_notification,
                min_gap_hours=min_gap_hours
            )
            penalties.append(penalty_recency)

        final_penalty = np.prod(penalties)
        return float(final_penalty)

    def _classify_engagement(self, user_features: UserFeatures) -> str:
        """
        Classify user engagement level.

        Args:
            user_features: User feature object

        Returns:
            'high', 'medium', or 'low'
        """
        # Need enough history to classify
        if user_features.notif_count_30d < 10:
            return 'medium'

        open_rate = user_features.open_rate_30d

        if open_rate >= 0.15:  # 15%+ open rate
            return 'high'
        elif open_rate >= 0.05:  # 5-15% open rate
            return 'medium'
        else:
            return 'low'
