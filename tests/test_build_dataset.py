"""Tests for dataset building."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import numpy as np

from src.features.build_dataset import (
    load_raw_events,
    add_template_features,
    create_model_features,
    build_training_dataset,
    get_feature_columns
)
from src.features.user_stats import attach_user_features
from src.config import Config


def create_raw_test_data():
    """Create raw test events."""
    events = []
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    for user_id in range(5):
        for i in range(20):
            events.append({
                'user_id': user_id,
                'template_id': f'template_{i % 5}',
                'timestamp_send': base_time + timedelta(hours=user_id*100 + i*2),
                'channel': 'push' if i % 2 == 0 else 'email',
                'notification_type': 'promo' if i % 3 == 0 else 'product_tip',
                'clicked': i % 4 == 0,
                'unsubscribed': False
            })

    return pd.DataFrame(events)


def test_load_raw_events():
    """Test loading raw events from file."""
    df = create_raw_test_data()

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)

        loaded = load_raw_events(Path(f.name))

        assert len(loaded) == len(df)
        assert 'user_id' in loaded.columns
        assert 'timestamp_send' in loaded.columns
        assert loaded['clicked'].dtype == bool


def test_add_template_features():
    """Test template feature encoding."""
    df = create_raw_test_data()

    result = add_template_features(df)

    assert 'channel_encoded' in result.columns
    assert 'notif_type_encoded' in result.columns
    assert 'template_hash' in result.columns

    # Check encodings are integers
    assert result['channel_encoded'].dtype in [np.int64, np.int32, np.int16]
    assert result['notif_type_encoded'].dtype in [np.int64, np.int32, np.int16]


def test_create_model_features():
    """Test creation of model features."""
    df = create_raw_test_data()
    df = add_template_features(df)
    df = attach_user_features(df)

    result = create_model_features(df)

    # Check derived features exist
    assert 'is_weekend' in result.columns
    assert 'is_night' in result.columns

    # Check they are binary
    assert set(result['is_weekend'].unique()).issubset({0, 1})
    assert set(result['is_night'].unique()).issubset({0, 1})


def test_build_training_dataset_integration():
    """Integration test for full dataset building pipeline."""
    df = create_raw_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save raw data
        raw_path = Path(tmpdir) / "raw_events.parquet"
        df.to_parquet(raw_path)

        # Build dataset
        output_path = Path(tmpdir) / "processed.parquet"
        train_df, test_df = build_training_dataset(
            raw_path, output_path, test_split=0.2
        )

        # Check split
        total = len(train_df) + len(test_df)
        assert total == len(df)
        assert len(test_df) / total >= 0.15  # Approximately 20%

        # Check train comes before test (temporal split)
        assert train_df['timestamp_send'].max() <= test_df['timestamp_send'].min()

        # Check files were saved
        train_path = Path(tmpdir) / "processed_train.parquet"
        test_path = Path(tmpdir) / "processed_test.parquet"
        assert train_path.exists()
        assert test_path.exists()


def test_get_feature_columns():
    """Test feature column list retrieval."""
    cols = get_feature_columns()

    assert isinstance(cols, list)
    assert len(cols) > 10  # Should have many features
    assert 'notif_count_24h' in cols
    assert 'open_rate_168h' in cols


def test_no_data_leakage():
    """Test that train and test sets don't overlap."""
    df = create_raw_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.parquet"
        df.to_parquet(raw_path)

        output_path = Path(tmpdir) / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path)

        # Train should come before test (temporal split)
        assert train_df['timestamp_send'].max() <= test_df['timestamp_send'].min()


def test_feature_consistency():
    """Test that all features are valid numbers."""
    df = create_raw_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.parquet"
        df.to_parquet(raw_path)

        output_path = Path(tmpdir) / "proc.parquet"
        train_df, _ = build_training_dataset(raw_path, output_path)

        feature_cols = get_feature_columns()

        for col in feature_cols:
            assert col in train_df.columns, f"Missing feature: {col}"
            # No NaN values after processing
            assert not train_df[col].isna().any(), f"NaN values in {col}"
            # All should be numeric
            assert pd.api.types.is_numeric_dtype(train_df[col]), f"{col} not numeric"
