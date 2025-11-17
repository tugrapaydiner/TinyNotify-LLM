"""
Build training dataset from raw events.
Converts raw notification events into supervised learning format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from src.config import Config, DEFAULT_CONFIG
from src.features.user_stats import attach_user_features
from src.features.schema import Channel, NotificationType


def load_raw_events(filepath: Path) -> pd.DataFrame:
    """
    Load raw events from file.

    Expected columns: user_id, template_id, timestamp_send, channel,
                      notification_type, clicked, unsubscribed
    """
    # Support multiple formats
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath, parse_dates=['timestamp_send'])
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Validate required columns
    required_cols = ['user_id', 'template_id', 'timestamp_send', 'channel']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Ensure proper types
    df['timestamp_send'] = pd.to_datetime(df['timestamp_send'])

    # Fill missing values for outcome columns
    if 'clicked' not in df.columns:
        df['clicked'] = False
    if 'unsubscribed' not in df.columns:
        df['unsubscribed'] = False

    df['clicked'] = df['clicked'].fillna(False).astype(bool)
    df['unsubscribed'] = df['unsubscribed'].fillna(False).astype(bool)

    return df


def add_template_features(
    events_df: pd.DataFrame,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """
    Add template-level features (channel encoding, type encoding, etc.).
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = events_df.copy()

    # Encode channel
    channel_map = {ch: idx for idx, ch in enumerate(config.features.channels)}
    df['channel_encoded'] = df['channel'].map(
        lambda x: channel_map.get(x, 0) if isinstance(x, str) else channel_map.get(x.value, 0)
    )

    # Encode notification type (if present)
    if 'notification_type' in df.columns:
        type_map = {t: idx for idx, t in enumerate(config.features.notification_types)}
        df['notif_type_encoded'] = df['notification_type'].map(
            lambda x: type_map.get(x, 0) if isinstance(x, str) else type_map.get(x.value, 0)
        )
    else:
        df['notif_type_encoded'] = 0

    # Hash template_id to fixed dimension
    df['template_hash'] = df['template_id'].apply(
        lambda x: hash(x) % config.features.template_hash_dim
    )

    return df


def create_model_features(
    events_df: pd.DataFrame,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """
    Create final feature matrix for model training.

    Returns DataFrame with all features needed for click/complaint models.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Should already have user features and template features attached
    df = events_df.copy()

    # Select feature columns
    feature_cols = [
        # User features
        'notif_count_24h', 'notif_count_168h', 'notif_count_720h',
        'click_count_168h', 'click_count_720h',
        'open_rate_168h', 'open_rate_720h',

        # Template features
        'channel_encoded', 'notif_type_encoded', 'template_hash',

        # Temporal features
        'hour_of_day', 'day_of_week',
    ]

    # Add derived features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] >= 22)).astype(int)

    feature_cols.extend(['is_weekend', 'is_night'])

    # Handle None values in hours_since columns
    df['hours_since_last_notif_168h'] = df.get('hours_since_last_notif_168h', np.nan).fillna(999)
    df['hours_since_last_click_168h'] = df.get('hours_since_last_click_168h', np.nan).fillna(999)

    feature_cols.extend(['hours_since_last_notif_168h', 'hours_since_last_click_168h'])

    # Ensure all feature columns exist and fill NaN values
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            # Fill NaN with 0 for count and rate columns
            if 'count' in col or 'rate' in col:
                df[col] = df[col].fillna(0)

    return df


def build_training_dataset(
    raw_events_path: Path,
    output_path: Path,
    config: Optional[Config] = None,
    test_split: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to build complete training dataset.

    Args:
        raw_events_path: Path to raw events file
        output_path: Path to save processed dataset
        config: Configuration
        test_split: Fraction of data to reserve for testing

    Returns:
        Tuple of (train_df, test_df)
    """
    if config is None:
        config = DEFAULT_CONFIG

    print("Loading raw events...")
    df = load_raw_events(raw_events_path)
    print(f"Loaded {len(df)} events for {df['user_id'].nunique()} users")

    print("\nAdding template features...")
    df = add_template_features(df, config)

    print("\nAttaching user features...")
    df = attach_user_features(df, config)

    print("\nCreating model features...")
    df = create_model_features(df, config)

    # Sort by timestamp for proper train/test split
    df = df.sort_values('timestamp_send').reset_index(drop=True)

    # Temporal split (not random - later events are test set)
    split_idx = int(len(df) * (1 - test_split))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTrain set: {len(train_df)} events")
    print(f"Test set: {len(test_df)} events")

    # Save
    train_path = output_path.parent / f"{output_path.stem}_train.parquet"
    test_path = output_path.parent / f"{output_path.stem}_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved train to: {train_path}")
    print(f"Saved test to: {test_path}")

    return train_df, test_df


def get_feature_columns(config: Optional[Config] = None) -> list:
    """
    Return list of feature column names used for modeling.
    """
    if config is None:
        config = DEFAULT_CONFIG

    return [
        'notif_count_24h', 'notif_count_168h', 'notif_count_720h',
        'click_count_168h', 'click_count_720h',
        'open_rate_168h', 'open_rate_720h',
        'channel_encoded', 'notif_type_encoded', 'template_hash',
        'hour_of_day', 'day_of_week',
        'is_weekend', 'is_night',
        'hours_since_last_notif_168h', 'hours_since_last_click_168h'
    ]
