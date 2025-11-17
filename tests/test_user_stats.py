"""Tests for user statistics computation."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.user_stats import (
    compute_time_window_stats,
    compute_open_rates,
    compute_hour_bucket,
    compute_most_active_hour_bucket,
    attach_user_features,
    build_user_features_object
)
from src.features.schema import UserFeatures
from src.config import Config


def create_sample_events(n_users=3, events_per_user=10):
    """Create sample event data for testing."""
    events = []
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    for user_id in range(n_users):
        for i in range(events_per_user):
            events.append({
                'user_id': user_id,
                'template_id': f'template_{i % 3}',
                'timestamp_send': base_time + timedelta(hours=i*6),
                'hour_of_day': (12 + i*6) % 24,
                'clicked': i % 3 == 0,  # Every 3rd event is clicked
                'unsubscribed': i == events_per_user - 1 and user_id == 0,  # Last event for user 0
            })

    return pd.DataFrame(events)


def test_time_window_stats_basic():
    """Test basic time window statistics computation."""
    df = create_sample_events(n_users=2, events_per_user=5)

    result = compute_time_window_stats(df, window_hours=24)

    # Check that new columns were added
    assert 'notif_count_24h' in result.columns
    assert 'click_count_24h' in result.columns
    assert 'hours_since_last_notif_24h' in result.columns

    # First event for each user should have 0 past notifications
    for user_id in df['user_id'].unique():
        user_first = result[result['user_id'] == user_id].iloc[0]
        assert user_first['notif_count_24h'] == 0


def test_time_window_stats_accumulation():
    """Test that window statistics accumulate correctly."""
    events = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    # Create events every 1 hour for same user
    for i in range(5):
        events.append({
            'user_id': 1,
            'template_id': 't1',
            'timestamp_send': base_time + timedelta(hours=i),
            'clicked': False,
            'unsubscribed': False
        })

    df = pd.DataFrame(events)
    result = compute_time_window_stats(df, window_hours=3)

    # Event 0: no past events (count=0)
    # Event 1: 1 past event (count=1)
    # Event 2: 2 past events (count=2)
    # Event 3: 3 past events (count=3)
    # Event 4: past events from hour 1,2,3 (count=3)

    counts = result['notif_count_3h'].tolist()
    assert counts[0] == 0
    assert counts[1] == 1
    assert counts[2] == 2
    assert counts[3] == 3
    assert counts[4] == 3  # Window slides


def test_open_rates_calculation():
    """Test open rate calculation."""
    df = pd.DataFrame({
        'user_id': [1, 1, 1],
        'notif_count_7h': [5, 10, 0],
        'click_count_7h': [1, 5, 0]
    })

    result = compute_open_rates(df, window_hours=7)

    assert 'open_rate_7h' in result.columns
    assert result.loc[0, 'open_rate_7h'] == 0.2  # 1/5
    assert result.loc[1, 'open_rate_7h'] == 0.5  # 5/10
    assert result.loc[2, 'open_rate_7h'] == 0.0  # 0/0 -> 0


def test_hour_bucket_conversion():
    """Test hour to bucket conversion."""
    assert compute_hour_bucket(2) == 'night'
    assert compute_hour_bucket(8) == 'morning'
    assert compute_hour_bucket(14) == 'afternoon'
    assert compute_hour_bucket(20) == 'evening'
    assert compute_hour_bucket(23) == 'evening'
    assert compute_hour_bucket(0) == 'night'


def test_most_active_hour_bucket():
    """Test finding user's most active hour bucket."""
    df = pd.DataFrame({
        'user_id': [1, 1, 1, 1, 1],
        'hour_of_day': [8, 9, 10, 14, 15],
        'clicked': [True, True, True, False, True]
    })

    most_active = compute_most_active_hour_bucket(df, user_id=1)

    # 3 clicks in morning (8,9,10), 1 in afternoon (15)
    assert most_active == 'morning'


def test_most_active_hour_bucket_no_clicks():
    """Test most active hour bucket when user has no clicks."""
    df = pd.DataFrame({
        'user_id': [1, 1, 1],
        'hour_of_day': [8, 9, 10],
        'clicked': [False, False, False]
    })

    most_active = compute_most_active_hour_bucket(df, user_id=1)
    assert most_active is None


def test_attach_user_features_integration():
    """Integration test for full feature attachment."""
    df = create_sample_events(n_users=2, events_per_user=10)

    result = attach_user_features(df)

    # Check all expected columns exist
    expected_cols = [
        'notif_count_24h', 'notif_count_168h', 'notif_count_720h',
        'click_count_168h', 'click_count_720h',
        'open_rate_168h', 'open_rate_720h',
        'hour_bucket'
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    # Check data types
    assert result['notif_count_24h'].dtype in [np.int64, np.int32, np.int16]
    assert result['open_rate_168h'].dtype in [np.float64, np.float32]


def test_build_user_features_object():
    """Test converting DataFrame row to UserFeatures object."""
    df = create_sample_events(n_users=1, events_per_user=5)
    df = attach_user_features(df)

    row = df.iloc[2]  # 3rd event

    user_features = build_user_features_object(row)

    assert isinstance(user_features, UserFeatures)
    assert user_features.user_id == row['user_id']
    assert user_features.notif_count_24h >= 0
    assert 0.0 <= user_features.open_rate_7d <= 1.0


def test_feature_invariants():
    """Test that computed features maintain invariants."""
    df = create_sample_events(n_users=3, events_per_user=20)
    result = attach_user_features(df)

    # For each row, check invariants
    for idx, row in result.iterrows():
        # Time window monotonicity
        assert row['notif_count_24h'] <= row['notif_count_168h']
        assert row['notif_count_168h'] <= row['notif_count_720h']

        # Open rates should be probabilities
        assert 0.0 <= row['open_rate_168h'] <= 1.0
        assert 0.0 <= row['open_rate_720h'] <= 1.0

        # Counts should be non-negative
        assert row['notif_count_24h'] >= 0
        assert row['click_count_168h'] >= 0
