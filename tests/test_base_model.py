"""Tests for base prediction models."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time

from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.features.build_dataset import (
    add_template_features,
    create_model_features,
    get_feature_columns
)
from src.features.user_stats import attach_user_features
from src.config import DEFAULT_CONFIG


def create_synthetic_dataset(n_samples: int = 1000, click_rate: float = 0.15):
    """Create synthetic dataset for testing."""
    np.random.seed(42)

    events = []
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    for i in range(n_samples):
        user_id = i % 100  # 100 unique users

        # Generate features that correlate with click
        hour_of_day = np.random.randint(0, 24)
        is_morning = 1 if 6 <= hour_of_day < 12 else 0

        # Click is more likely in morning
        p_click = click_rate * (1.5 if is_morning else 0.7)
        clicked = np.random.random() < p_click

        # Rare complaints
        unsubscribed = np.random.random() < 0.02

        events.append({
            'user_id': user_id,
            'template_id': f'template_{i % 10}',
            'timestamp_send': base_time + timedelta(hours=i),
            'channel': 'push' if i % 2 == 0 else 'email',
            'notification_type': 'promo' if i % 3 == 0 else 'product_tip',
            'clicked': clicked,
            'unsubscribed': unsubscribed
        })

    df = pd.DataFrame(events)

    # Add features
    df = add_template_features(df)
    df = attach_user_features(df)
    df = create_model_features(df)

    return df


def test_click_model_training():
    """Test click model training."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    metrics = model.train(df, target_col='clicked', verbose=False)

    # Check model was trained
    assert model.model is not None

    # Check metrics exist
    assert 'train_auc' in metrics
    assert 'train_logloss' in metrics

    # AUC should be reasonable (better than random)
    assert metrics['train_auc'] > 0.5


def test_click_model_training_with_validation():
    """Test click model training with validation set."""
    df = create_synthetic_dataset(n_samples=800)

    # Split into train/valid
    train_df = df.iloc[:600].copy()
    valid_df = df.iloc[600:].copy()

    model = BaseClickModel()
    metrics = model.train(train_df, target_col='clicked', valid_df=valid_df, verbose=False)

    # Check validation metrics exist
    assert 'valid_auc' in metrics
    assert 'valid_logloss' in metrics

    # Validation AUC should be reasonable
    assert metrics['valid_auc'] > 0.5


def test_click_model_prediction():
    """Test click model prediction."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Predict on same data
    predictions = model.predict(df)

    # Check predictions
    assert len(predictions) == len(df)
    assert all(0 <= p <= 1 for p in predictions)  # Valid probabilities


def test_click_model_single_prediction():
    """Test single instance prediction."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Create feature dict
    features = {col: 0.0 for col in get_feature_columns()}
    features['notif_count_24h'] = 2
    features['open_rate_168h'] = 0.15
    features['hour_of_day'] = 10

    # Predict
    prob = model.predict_single(features)

    # Check probability is valid
    assert 0 <= prob <= 1


def test_click_model_save_load():
    """Test model save and load."""
    df = create_synthetic_dataset(n_samples=500)

    # Train model
    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Get predictions before save
    preds_before = model.predict(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "click_model"

        # Save
        model.save(model_path)

        # Check files exist
        assert (Path(tmpdir) / "click_model.txt").exists()
        assert (Path(tmpdir) / "click_model.json").exists()

        # Load into new model
        model2 = BaseClickModel()
        model2.load(model_path)

        # Get predictions after load
        preds_after = model2.predict(df)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_click_model_feature_importance():
    """Test feature importance extraction."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Get feature importance
    importance_df = model.get_feature_importance()

    # Check structure
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert len(importance_df) == len(get_feature_columns())

    # Check sorted descending
    assert importance_df['importance'].is_monotonic_decreasing


def test_complaint_model_training():
    """Test complaint model training."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseComplaintModel()
    metrics = model.train(df, target_col='unsubscribed', verbose=False)

    # Check model was trained
    assert model.model is not None

    # Check metrics exist
    assert 'train_auc' in metrics or 'train_logloss' in metrics

    # Check scale_pos_weight was computed
    assert 'scale_pos_weight' in model.training_metadata
    assert model.training_metadata['scale_pos_weight'] > 1  # Complaints are rare


def test_complaint_model_prediction():
    """Test complaint model prediction."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseComplaintModel()
    model.train(df, target_col='unsubscribed', verbose=False)

    # Predict
    predictions = model.predict(df)

    # Check predictions
    assert len(predictions) == len(df)
    assert all(0 <= p <= 1 for p in predictions)


def test_complaint_model_save_load():
    """Test complaint model save and load."""
    df = create_synthetic_dataset(n_samples=500)

    # Train model
    model = BaseComplaintModel()
    model.train(df, target_col='unsubscribed', verbose=False)

    # Get predictions before save
    preds_before = model.predict(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "complaint_model"

        # Save
        model.save(model_path)

        # Load into new model
        model2 = BaseComplaintModel()
        model2.load(model_path)

        # Get predictions after load
        preds_after = model2.predict(df)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_model_inference_speed():
    """Test that model inference is fast enough."""
    df = create_synthetic_dataset(n_samples=1000)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Measure inference time for batch
    start = time.time()
    _ = model.predict(df)
    elapsed_ms = (time.time() - start) * 1000

    # Should be fast (< 500ms for 1000 samples as per config)
    assert elapsed_ms < 500, f"Inference took {elapsed_ms:.2f}ms, expected < 500ms"


def test_model_memory_efficiency():
    """Test that models don't use excessive memory."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Train both models
    df = create_synthetic_dataset(n_samples=1000)

    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)

    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    # Predict
    _ = click_model.predict(df)
    _ = complaint_model.predict(df)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before

    # Memory increase should be reasonable (< 100 MB)
    assert mem_used < 100, f"Models used {mem_used:.2f} MB, expected < 100 MB"


def test_model_handles_missing_features():
    """Test that models handle missing features gracefully."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    # Create incomplete feature dict
    features = {'hour_of_day': 10}  # Only one feature

    # Should still predict (filling missing with 0)
    prob = model.predict_single(features)
    assert 0 <= prob <= 1


def test_model_probability_calibration():
    """Test that predicted probabilities are reasonably calibrated."""
    df = create_synthetic_dataset(n_samples=1000, click_rate=0.2)

    # Split
    train_df = df.iloc[:800].copy()
    test_df = df.iloc[800:].copy()

    model = BaseClickModel()
    model.train(train_df, target_col='clicked', verbose=False)

    # Predict on test set
    preds = model.predict(test_df)

    # Mean prediction should be close to actual click rate
    actual_rate = test_df['clicked'].mean()
    pred_rate = preds.mean()

    # Should be within 10% of actual
    assert abs(pred_rate - actual_rate) < 0.1, \
        f"Predicted rate {pred_rate:.3f} too far from actual {actual_rate:.3f}"


def test_training_metadata():
    """Test that training metadata is captured correctly."""
    df = create_synthetic_dataset(n_samples=500)

    model = BaseClickModel()
    model.train(df, target_col='clicked', verbose=False)

    metadata = model.training_metadata

    # Check required fields
    assert 'train_timestamp' in metadata
    assert 'n_train_samples' in metadata
    assert 'n_features' in metadata
    assert 'target_positive_rate' in metadata

    # Check values
    assert metadata['n_train_samples'] == len(df)
    assert metadata['n_features'] == len(get_feature_columns())
    assert 0 < metadata['target_positive_rate'] < 1


def test_model_with_zero_variance_features():
    """Test model handles features with zero variance."""
    df = create_synthetic_dataset(n_samples=500)

    # Add zero-variance feature
    df['is_weekend'] = 0  # All same value

    model = BaseClickModel()

    # Should train without error
    metrics = model.train(df, target_col='clicked', verbose=False)
    assert 'train_auc' in metrics
