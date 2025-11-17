"""
Data schemas for TinyNotify-LLM.
Defines the structure of users, notifications, and events.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum


class Channel(str, Enum):
    """Notification channel types."""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    IN_APP = "in_app"


class NotificationType(str, Enum):
    """Types of notifications."""
    PROMO = "promo"
    PRODUCT_TIP = "product_tip"
    IMPORTANT = "important"
    SYSTEM_ALERT = "system_alert"


@dataclass
class User:
    """User entity."""
    user_id: int
    device_type: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = "en"
    timezone_offset: int = 0  # Hours from UTC
    created_at: Optional[datetime] = None

    def __post_init__(self):
        assert self.user_id >= 0, "user_id must be non-negative"
        assert -12 <= self.timezone_offset <= 14, "Invalid timezone offset"


@dataclass
class NotificationTemplate:
    """Notification template entity."""
    template_id: str
    channel: Channel
    notification_type: NotificationType
    text: str
    subject: Optional[str] = None  # For emails

    def __post_init__(self):
        assert len(self.template_id) > 0, "template_id cannot be empty"
        assert len(self.text) > 0, "text cannot be empty"
        assert len(self.text) <= 500, "text too long (max 500 chars)"

        if self.channel == Channel.EMAIL:
            assert self.subject is not None, "Email must have subject"


@dataclass
class NotificationEvent:
    """A notification send event with outcomes."""
    event_id: int
    user_id: int
    template_id: str
    timestamp_send: datetime
    channel: Channel
    notification_type: NotificationType

    # Outcomes (labels)
    clicked: bool = False
    opened: bool = False
    unsubscribed: bool = False

    # Optional metadata
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None

    def __post_init__(self):
        assert self.event_id >= 0
        assert self.user_id >= 0

        # Derive hour and day if not provided
        if self.hour_of_day is None:
            self.hour_of_day = self.timestamp_send.hour
        if self.day_of_week is None:
            self.day_of_week = self.timestamp_send.weekday()

        assert 0 <= self.hour_of_day <= 23
        assert 0 <= self.day_of_week <= 6

        # If clicked, must have been opened
        if self.clicked:
            self.opened = True


@dataclass
class UserFeatures:
    """
    Computed features for a user at a specific point in time.
    These are time-dependent rolling statistics.
    """
    user_id: int
    timestamp: datetime

    # Notification volume features
    notif_count_24h: int = 0
    notif_count_7d: int = 0
    notif_count_30d: int = 0

    # Engagement features
    click_count_7d: int = 0
    click_count_30d: int = 0
    open_rate_7d: float = 0.0
    open_rate_30d: float = 0.0

    # Recency features
    hours_since_last_notification: Optional[float] = None
    hours_since_last_click: Optional[float] = None

    # Temporal patterns
    most_active_hour_bucket: Optional[str] = None  # morning/afternoon/evening/night

    # Fatigue signals
    complaint_count_30d: int = 0

    def __post_init__(self):
        assert self.user_id >= 0
        assert self.notif_count_24h <= self.notif_count_7d
        assert self.notif_count_7d <= self.notif_count_30d
        assert 0.0 <= self.open_rate_7d <= 1.0
        assert 0.0 <= self.open_rate_30d <= 1.0


@dataclass
class CandidateNotification:
    """
    A candidate notification to be scored.
    Combines notification template with user features and context.
    """
    user_id: int
    template_id: str
    channel: Channel
    notification_type: NotificationType
    text: str

    # Context
    timestamp: datetime
    hour_of_day: int
    day_of_week: int

    # User features at decision time
    user_features: UserFeatures

    # Optional: if this is from historical data, include ground truth
    ground_truth_clicked: Optional[bool] = None
    ground_truth_unsubscribed: Optional[bool] = None

    def __post_init__(self):
        assert 0 <= self.hour_of_day <= 23
        assert 0 <= self.day_of_week <= 6
        assert self.user_features.user_id == self.user_id


@dataclass
class ScoredNotification:
    """
    A candidate notification with all computed scores.
    """
    candidate: CandidateNotification

    # Model predictions
    p_click: float
    p_complaint: float
    base_score: float

    # LLM rating (if computed)
    llm_rating: Optional[int] = None  # 1-5
    llm_rating_normalized: Optional[float] = None  # 0-1

    # Fatigue penalty applied
    fatigue_penalty: float = 1.0

    # Final combined score
    final_score: Optional[float] = None

    def __post_init__(self):
        assert 0.0 <= self.p_click <= 1.0
        assert 0.0 <= self.p_complaint <= 1.0

        if self.llm_rating is not None:
            assert 1 <= self.llm_rating <= 5
            if self.llm_rating_normalized is None:
                self.llm_rating_normalized = (self.llm_rating - 1) / 4.0

        if self.llm_rating_normalized is not None:
            assert 0.0 <= self.llm_rating_normalized <= 1.0


@dataclass
class Decision:
    """
    Final decision for a notification.
    """
    candidate: CandidateNotification
    scored: ScoredNotification

    # Decision
    should_send: bool

    # Reason (for debugging/explanation)
    reason: str

    # Timestamp of decision
    decision_timestamp: datetime

    def __post_init__(self):
        assert self.candidate == self.scored.candidate
