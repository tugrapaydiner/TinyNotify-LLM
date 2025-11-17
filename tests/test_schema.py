"""Tests for data schemas."""
import pytest
from datetime import datetime, timedelta

from src.features.schema import (
    User, NotificationTemplate, NotificationEvent, UserFeatures,
    CandidateNotification, ScoredNotification, Decision,
    Channel, NotificationType
)


def test_user_creation():
    """Test User dataclass creation and validation."""
    user = User(user_id=12345, country="US", language="en")
    assert user.user_id == 12345
    assert user.country == "US"

    # Test validation
    with pytest.raises(AssertionError):
        User(user_id=-1)  # Negative ID

    with pytest.raises(AssertionError):
        User(user_id=1, timezone_offset=20)  # Invalid timezone


def test_notification_template_creation():
    """Test NotificationTemplate validation."""
    template = NotificationTemplate(
        template_id="promo_001",
        channel=Channel.EMAIL,
        notification_type=NotificationType.PROMO,
        text="Check out our sale!",
        subject="Big Sale Today"
    )
    assert template.template_id == "promo_001"

    # Email without subject should fail
    with pytest.raises(AssertionError):
        NotificationTemplate(
            template_id="test",
            channel=Channel.EMAIL,
            notification_type=NotificationType.PROMO,
            text="Test"
        )

    # Text too long should fail
    with pytest.raises(AssertionError):
        NotificationTemplate(
            template_id="test",
            channel=Channel.PUSH,
            notification_type=NotificationType.PROMO,
            text="x" * 501
        )


def test_notification_event_invariants():
    """Test NotificationEvent maintains invariants."""
    event = NotificationEvent(
        event_id=1,
        user_id=123,
        template_id="template_1",
        timestamp_send=datetime.now(),
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        clicked=True
    )

    # If clicked, opened should be automatically set to True
    assert event.opened == True

    # Hour and day should be derived
    assert 0 <= event.hour_of_day <= 23
    assert 0 <= event.day_of_week <= 6


def test_user_features_invariants():
    """Test UserFeatures maintains statistical invariants."""
    features = UserFeatures(
        user_id=123,
        timestamp=datetime.now(),
        notif_count_24h=5,
        notif_count_7d=20,
        notif_count_30d=50,
        click_count_7d=3,
        open_rate_7d=0.15
    )

    # Time windows should be monotonic
    assert features.notif_count_24h <= features.notif_count_7d
    assert features.notif_count_7d <= features.notif_count_30d

    # Open rates should be valid probabilities
    assert 0.0 <= features.open_rate_7d <= 1.0

    # Test invalid open rate
    with pytest.raises(AssertionError):
        UserFeatures(
            user_id=123,
            timestamp=datetime.now(),
            open_rate_7d=1.5  # > 1.0
        )


def test_candidate_notification_consistency():
    """Test CandidateNotification maintains user_id consistency."""
    user_features = UserFeatures(
        user_id=123,
        timestamp=datetime.now(),
        notif_count_24h=2,
        notif_count_7d=5,
        notif_count_30d=10
    )

    candidate = CandidateNotification(
        user_id=123,
        template_id="t1",
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text="Tip: ...",
        timestamp=datetime.now(),
        hour_of_day=14,
        day_of_week=2,
        user_features=user_features
    )

    assert candidate.user_id == candidate.user_features.user_id

    # Mismatched user_id should fail
    with pytest.raises(AssertionError):
        CandidateNotification(
            user_id=999,  # Different from user_features
            template_id="t1",
            channel=Channel.PUSH,
            notification_type=NotificationType.PRODUCT_TIP,
            text="Tip: ...",
            timestamp=datetime.now(),
            hour_of_day=14,
            day_of_week=2,
            user_features=user_features
        )


def test_scored_notification_probability_ranges():
    """Test ScoredNotification validates probability ranges."""
    user_features = UserFeatures(user_id=1, timestamp=datetime.now())
    candidate = CandidateNotification(
        user_id=1,
        template_id="t1",
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text="Test",
        timestamp=datetime.now(),
        hour_of_day=12,
        day_of_week=1,
        user_features=user_features
    )

    scored = ScoredNotification(
        candidate=candidate,
        p_click=0.3,
        p_complaint=0.05,
        base_score=0.25,
        llm_rating=4
    )

    assert 0.0 <= scored.p_click <= 1.0
    assert 0.0 <= scored.p_complaint <= 1.0
    assert scored.llm_rating_normalized == 0.75  # (4-1)/4

    # Test invalid probabilities
    with pytest.raises(AssertionError):
        ScoredNotification(
            candidate=candidate,
            p_click=1.5,  # > 1.0
            p_complaint=0.05,
            base_score=0.0
        )


def test_decision_consistency():
    """Test Decision maintains consistency between components."""
    user_features = UserFeatures(user_id=1, timestamp=datetime.now())
    candidate = CandidateNotification(
        user_id=1,
        template_id="t1",
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text="Test",
        timestamp=datetime.now(),
        hour_of_day=12,
        day_of_week=1,
        user_features=user_features
    )

    scored = ScoredNotification(
        candidate=candidate,
        p_click=0.3,
        p_complaint=0.05,
        base_score=0.25
    )

    decision = Decision(
        candidate=candidate,
        scored=scored,
        should_send=True,
        reason="Score above threshold",
        decision_timestamp=datetime.now()
    )

    assert decision.candidate == decision.scored.candidate
