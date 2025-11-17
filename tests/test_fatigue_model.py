"""Tests for fatigue model."""
import pytest
import numpy as np
from datetime import datetime

from src.models.fatigue_model import FatigueModel, AdaptiveFatigueModel
from src.features.schema import UserFeatures
from src.config import DEFAULT_CONFIG


def create_user_features(
    notif_count_24h: int = 0,
    notif_count_7d: int = 0,
    notif_count_30d: int = 0,
    open_rate_7d: float = 0.1,
    open_rate_30d: float = 0.1,
    hours_since_last: float = None
) -> UserFeatures:
    """Helper to create UserFeatures for testing."""
    return UserFeatures(
        user_id=1,
        timestamp=datetime.now(),
        notif_count_24h=notif_count_24h,
        notif_count_7d=max(notif_count_7d, notif_count_24h),  # Ensure monotonicity
        notif_count_30d=max(notif_count_30d, notif_count_7d),
        open_rate_7d=open_rate_7d,
        open_rate_30d=open_rate_30d,
        hours_since_last_notification=hours_since_last
    )


def test_fatigue_model_initialization():
    """Test fatigue model initialization."""
    model = FatigueModel()
    assert model.config is not None


def test_no_fatigue_for_new_user():
    """Test that new users with no history have no fatigue."""
    model = FatigueModel()
    user = create_user_features(
        notif_count_24h=0,
        notif_count_7d=0,
        notif_count_30d=0
    )

    penalty = model.compute_fatigue_penalty(user)

    # No fatigue for new user
    assert penalty == 1.0


def test_no_fatigue_for_low_volume():
    """Test no fatigue for users within normal limits."""
    model = FatigueModel()
    user = create_user_features(
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30,
        open_rate_7d=0.15
    )

    penalty = model.compute_fatigue_penalty(user)

    # Should be no penalty (penalty = 1.0)
    assert penalty == 1.0


def test_volume_penalty_24h():
    """Test penalty for high 24h notification volume."""
    model = FatigueModel()

    # Within limit
    user1 = create_user_features(notif_count_24h=3, notif_count_7d=10, notif_count_30d=30)
    penalty1 = model.compute_fatigue_penalty(user1)

    # Exceeds limit (default max_daily is 5)
    user2 = create_user_features(notif_count_24h=8, notif_count_7d=20, notif_count_30d=50)
    penalty2 = model.compute_fatigue_penalty(user2)

    # Higher volume should have lower penalty
    assert penalty2 < penalty1


def test_volume_penalty_7d():
    """Test penalty for high 7d notification volume."""
    model = FatigueModel()

    user1 = create_user_features(notif_count_24h=3, notif_count_7d=20, notif_count_30d=50)
    penalty1 = model.compute_fatigue_penalty(user1)

    user2 = create_user_features(notif_count_24h=3, notif_count_7d=50, notif_count_30d=80)
    penalty2 = model.compute_fatigue_penalty(user2)

    # Higher 7d volume should have lower penalty
    assert penalty2 < penalty1


def test_engagement_penalty():
    """Test penalty for low engagement."""
    model = FatigueModel()

    # High engagement
    user1 = create_user_features(
        notif_count_7d=20,
        notif_count_30d=50,
        open_rate_7d=0.20
    )
    penalty1 = model.compute_fatigue_penalty(user1)

    # Low engagement
    user2 = create_user_features(
        notif_count_7d=20,
        notif_count_30d=50,
        open_rate_7d=0.02
    )
    penalty2 = model.compute_fatigue_penalty(user2)

    # Low engagement should have lower penalty
    assert penalty2 < penalty1


def test_engagement_penalty_requires_history():
    """Test that engagement penalty only applies with sufficient history."""
    model = FatigueModel()

    # Not enough history (< 5 notifications)
    user = create_user_features(
        notif_count_7d=3,
        notif_count_30d=10,
        open_rate_7d=0.0  # Zero open rate
    )

    penalty = model.compute_fatigue_penalty(user)

    # Should not be heavily penalized due to insufficient history
    assert penalty > 0.5


def test_recency_penalty():
    """Test penalty for notifications sent too soon."""
    model = FatigueModel()

    # Long time since last notification
    user1 = create_user_features(
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30,
        hours_since_last=5.0
    )
    penalty1 = model.compute_fatigue_penalty(user1)

    # Very recent notification
    user2 = create_user_features(
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30,
        hours_since_last=0.3  # 18 minutes ago
    )
    penalty2 = model.compute_fatigue_penalty(user2)

    # Recent notification should have lower penalty
    assert penalty2 < penalty1


def test_combined_penalties_are_multiplicative():
    """Test that multiple penalties combine multiplicatively."""
    model = FatigueModel()

    # User with multiple fatigue factors
    user = create_user_features(
        notif_count_24h=8,  # High volume
        notif_count_7d=40,
        notif_count_30d=80,
        open_rate_7d=0.02,  # Low engagement
        hours_since_last=0.5  # Recent notification
    )

    penalty = model.compute_fatigue_penalty(user)

    # Combined penalty should be quite low
    assert penalty < 0.5


def test_should_suppress():
    """Test suppression decision."""
    model = FatigueModel()

    # Healthy user
    user1 = create_user_features(
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30,
        open_rate_7d=0.15
    )
    assert not model.should_suppress(user1)

    # Fatigued user
    user2 = create_user_features(
        notif_count_24h=10,
        notif_count_7d=50,
        notif_count_30d=100,
        open_rate_7d=0.01,
        hours_since_last=0.2
    )
    assert model.should_suppress(user2)


def test_adjust_score():
    """Test score adjustment."""
    model = FatigueModel()

    base_score = 0.5

    # Healthy user - score should stay similar
    user1 = create_user_features(
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30
    )
    adjusted1 = model.adjust_score(base_score, user1)
    assert adjusted1 == base_score  # No penalty

    # Fatigued user - score should decrease
    user2 = create_user_features(
        notif_count_24h=8,
        notif_count_7d=40,
        notif_count_30d=80,
        open_rate_7d=0.02
    )
    adjusted2 = model.adjust_score(base_score, user2)
    assert adjusted2 < base_score


def test_get_fatigue_stats():
    """Test fatigue statistics reporting."""
    model = FatigueModel()

    user = create_user_features(
        notif_count_24h=5,
        notif_count_7d=20,
        notif_count_30d=50,
        open_rate_7d=0.10,
        hours_since_last=2.0
    )

    stats = model.get_fatigue_stats(user)

    # Check all expected keys exist
    assert 'overall_penalty' in stats
    assert 'volume_24h_penalty' in stats
    assert 'volume_7d_penalty' in stats
    assert 'engagement_penalty' in stats
    assert 'recency_penalty' in stats
    assert 'notif_count_24h' in stats
    assert 'open_rate_7d' in stats


def test_penalty_bounds():
    """Test that penalties are always in valid range."""
    model = FatigueModel()

    # Test extreme cases
    test_cases = [
        create_user_features(notif_count_24h=0, notif_count_7d=0, notif_count_30d=0),
        create_user_features(notif_count_24h=100, notif_count_7d=500, notif_count_30d=1000),
        create_user_features(notif_count_7d=50, notif_count_30d=100, open_rate_7d=0.0),
        create_user_features(notif_count_7d=50, notif_count_30d=100, open_rate_7d=1.0),
        create_user_features(notif_count_7d=10, notif_count_30d=30, hours_since_last=0.1),
        create_user_features(notif_count_7d=10, notif_count_30d=30, hours_since_last=100.0),
    ]

    for user in test_cases:
        penalty = model.compute_fatigue_penalty(user)
        assert 0.0 <= penalty <= 1.0, f"Penalty {penalty} out of bounds"


def test_adaptive_fatigue_model():
    """Test adaptive fatigue model."""
    model = AdaptiveFatigueModel()

    # High engagement user can receive more notifications
    high_engagement_user = create_user_features(
        notif_count_24h=8,  # Would normally trigger penalty
        notif_count_7d=40,
        notif_count_30d=100,
        open_rate_30d=0.25  # High engagement
    )

    # Low engagement user with same volume
    low_engagement_user = create_user_features(
        notif_count_24h=8,
        notif_count_7d=40,
        notif_count_30d=100,
        open_rate_30d=0.03  # Low engagement
    )

    penalty_high = model.compute_fatigue_penalty(high_engagement_user)
    penalty_low = model.compute_fatigue_penalty(low_engagement_user)

    # High engagement user should be penalized less
    assert penalty_high > penalty_low


def test_adaptive_model_classification():
    """Test engagement tier classification."""
    model = AdaptiveFatigueModel()

    # High engagement
    user1 = create_user_features(
        notif_count_30d=100,
        open_rate_30d=0.20
    )
    tier1 = model._classify_engagement(user1)
    assert tier1 == 'high'

    # Medium engagement
    user2 = create_user_features(
        notif_count_30d=100,
        open_rate_30d=0.08
    )
    tier2 = model._classify_engagement(user2)
    assert tier2 == 'medium'

    # Low engagement
    user3 = create_user_features(
        notif_count_30d=100,
        open_rate_30d=0.02
    )
    tier3 = model._classify_engagement(user3)
    assert tier3 == 'low'


def test_adaptive_model_insufficient_history():
    """Test adaptive model with insufficient history."""
    model = AdaptiveFatigueModel()

    # User with little history
    user = create_user_features(
        notif_count_30d=5,  # Too few
        open_rate_30d=0.0
    )

    tier = model._classify_engagement(user)

    # Should default to medium
    assert tier == 'medium'


def test_fatigue_penalty_monotonicity():
    """Test that fatigue increases with notification volume."""
    model = FatigueModel()

    penalties = []
    for count in [0, 3, 6, 9, 12, 15]:
        user = create_user_features(
            notif_count_24h=count,
            notif_count_7d=count * 3,
            notif_count_30d=count * 5
        )
        penalty = model.compute_fatigue_penalty(user)
        penalties.append(penalty)

    # Penalties should generally decrease as volume increases
    # (allowing for some variation due to multiple components)
    assert penalties[0] >= penalties[-1]


def test_zero_engagement_doesnt_block_completely():
    """Test that zero engagement doesn't completely block notifications."""
    model = FatigueModel()

    user = create_user_features(
        notif_count_7d=20,
        notif_count_30d=50,
        open_rate_7d=0.0  # No engagement
    )

    penalty = model.compute_fatigue_penalty(user)

    # Should still have some chance (>= 0.3 as per implementation)
    assert penalty >= 0.3
