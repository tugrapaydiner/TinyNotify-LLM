"""
User statistics computation.
Efficient rolling window calculations for user-level features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.config import Config, DEFAULT_CONFIG
from src.features.schema import UserFeatures, NotificationEvent


def compute_time_window_stats(
    events_df: pd.DataFrame,
    window_hours: int,
    timestamp_col: str = 'timestamp_send'
) -> pd.DataFrame:
    """
    For each event, compute statistics in the time window before it.

    Args:
        events_df: DataFrame with columns [user_id, timestamp_send, clicked, unsubscribed, ...]
        window_hours: Number of hours to look back
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with additional columns for window statistics
    """
    # Sort by user and time
    df = events_df.sort_values(['user_id', timestamp_col]).copy()

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    results = []

    for user_id in df['user_id'].unique():
        user_events = df[df['user_id'] == user_id].reset_index(drop=True)

        for idx in range(len(user_events)):
            current_time = user_events.loc[idx, timestamp_col]
            window_start = current_time - timedelta(hours=window_hours)

            # Get events in window (excluding current event)
            past_events = user_events[
                (user_events[timestamp_col] >= window_start) &
                (user_events[timestamp_col] < current_time)
            ]

            # Compute statistics
            stats = {
                'event_idx': user_events.index[idx],
                f'notif_count_{window_hours}h': len(past_events),
                f'click_count_{window_hours}h': past_events['clicked'].sum() if 'clicked' in past_events else 0,
                f'unsub_count_{window_hours}h': past_events.get('unsubscribed', pd.Series([0])).sum(),
            }

            # Hours since last notification
            if len(past_events) > 0:
                last_notif_time = past_events[timestamp_col].max()
                stats[f'hours_since_last_notif_{window_hours}h'] = \
                    (current_time - last_notif_time).total_seconds() / 3600
            else:
                stats[f'hours_since_last_notif_{window_hours}h'] = None

            # Hours since last click
            clicked_events = past_events[past_events.get('clicked', False)]
            if len(clicked_events) > 0:
                last_click_time = clicked_events[timestamp_col].max()
                stats[f'hours_since_last_click_{window_hours}h'] = \
                    (current_time - last_click_time).total_seconds() / 3600
            else:
                stats[f'hours_since_last_click_{window_hours}h'] = None

            results.append(stats)

    # Convert to DataFrame
    stats_df = pd.DataFrame(results)

    # Reset index on df to avoid duplicate index issues
    df = df.reset_index(drop=True)

    # Add new columns to dataframe (stats_df is already in same order as df)
    for col in stats_df.columns:
        if col != 'event_idx':
            df[col] = stats_df[col].values

    return df


def compute_open_rates(
    events_df: pd.DataFrame,
    window_hours: int
) -> pd.DataFrame:
    """
    Compute click/open rates for each event based on past window.

    Args:
        events_df: DataFrame with window statistics already computed
        window_hours: Window size in hours

    Returns:
        DataFrame with open_rate columns added
    """
    df = events_df.copy()

    notif_col = f'notif_count_{window_hours}h'
    click_col = f'click_count_{window_hours}h'
    rate_col = f'open_rate_{window_hours}h'

    # Compute rate, handling division by zero
    df[rate_col] = df.apply(
        lambda row: row[click_col] / row[notif_col]
                    if row[notif_col] > 0 else 0.0,
        axis=1
    )

    return df


def compute_hour_bucket(hour: int) -> str:
    """
    Convert hour of day (0-23) to time bucket.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time bucket name: 'night', 'morning', 'afternoon', 'evening'
    """
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def compute_most_active_hour_bucket(
    events_df: pd.DataFrame,
    user_id: int,
    window_hours: int = 168  # 7 days
) -> Optional[str]:
    """
    Find the time bucket where user clicks most frequently.

    Args:
        events_df: DataFrame with user events
        user_id: User ID to analyze
        window_hours: How far back to look

    Returns:
        Most active hour bucket or None if no clicks
    """
    user_events = events_df[events_df['user_id'] == user_id]
    clicked_events = user_events[user_events.get('clicked', False)]

    if len(clicked_events) == 0:
        return None

    # Get hour buckets for clicked events
    clicked_events = clicked_events.copy()
    clicked_events['hour_bucket'] = clicked_events['hour_of_day'].apply(compute_hour_bucket)

    # Count clicks per bucket
    bucket_counts = clicked_events['hour_bucket'].value_counts()

    if len(bucket_counts) == 0:
        return None

    return bucket_counts.idxmax()


def attach_user_features(
    events_df: pd.DataFrame,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """
    Main function to attach all user-level features to events.

    Args:
        events_df: DataFrame with raw notification events
        config: Configuration object (uses default if None)

    Returns:
        DataFrame with all user features attached
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = events_df.copy()

    # Ensure required columns exist
    required_cols = ['user_id', 'timestamp_send']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Add hour_of_day and day_of_week if not present
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['timestamp_send']).dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = pd.to_datetime(df['timestamp_send']).dt.dayofweek

    # Compute statistics for different time windows
    print("Computing 24h window statistics...")
    df = compute_time_window_stats(df, config.features.window_24h)

    print("Computing 7d window statistics...")
    df = compute_time_window_stats(df, config.features.window_7d)

    print("Computing 30d window statistics...")
    df = compute_time_window_stats(df, config.features.window_30d)

    # Compute open rates
    print("Computing open rates...")
    df = compute_open_rates(df, config.features.window_7d)
    df = compute_open_rates(df, config.features.window_30d)

    # Add hour bucket
    df['hour_bucket'] = df['hour_of_day'].apply(compute_hour_bucket)

    print("Feature attachment complete.")
    return df


def build_user_features_object(
    row: pd.Series,
    config: Optional[Config] = None
) -> UserFeatures:
    """
    Convert a DataFrame row with features into a UserFeatures object.

    Args:
        row: Pandas Series with feature columns
        config: Configuration object

    Returns:
        UserFeatures object
    """
    if config is None:
        config = DEFAULT_CONFIG

    return UserFeatures(
        user_id=int(row['user_id']),
        timestamp=row['timestamp_send'],
        notif_count_24h=int(row.get(f'notif_count_{config.features.window_24h}h', 0)),
        notif_count_7d=int(row.get(f'notif_count_{config.features.window_7d}h', 0)),
        notif_count_30d=int(row.get(f'notif_count_{config.features.window_30d}h', 0)),
        click_count_7d=int(row.get(f'click_count_{config.features.window_7d}h', 0)),
        click_count_30d=int(row.get(f'click_count_{config.features.window_30d}h', 0)),
        open_rate_7d=float(row.get(f'open_rate_{config.features.window_7d}h', 0.0)),
        open_rate_30d=float(row.get(f'open_rate_{config.features.window_30d}h', 0.0)),
        hours_since_last_notification=row.get(f'hours_since_last_notif_{config.features.window_7d}h'),
        hours_since_last_click=row.get(f'hours_since_last_click_{config.features.window_7d}h'),
        most_active_hour_bucket=row.get('hour_bucket'),
        complaint_count_30d=int(row.get(f'unsub_count_{config.features.window_30d}h', 0))
    )
