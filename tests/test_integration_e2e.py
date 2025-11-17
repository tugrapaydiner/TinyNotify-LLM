"""
End-to-end integration tests.
Tests the complete pipeline from data to decisions to API.
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pickle
import time

from src.config import DEFAULT_CONFIG
from src.features.build_dataset import build_training_dataset
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.scoring import NotificationScorer
from src.policy.decision import NotificationDecisionMaker
from src.simulation.simulator import NotificationSimulator
from src.simulation.metrics import MetricsCalculator, format_metrics_report


def create_synthetic_e2e_dataset(n_users=50, events_per_user=20):
    """Create synthetic dataset for end-to-end testing."""
    events = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for user_id in range(n_users):
        for i in range(events_per_user):
            # Create realistic temporal progression
            timestamp = base_time + timedelta(hours=user_id * 10 + i * 2)

            events.append({
                'user_id': user_id,
                'template_id': f'template_{i % 10}',
                'timestamp_send': timestamp,
                'channel': ['push', 'email', 'sms', 'in_app'][i % 4],
                'notification_type': ['promo', 'product_tip', 'important', 'system_alert'][i % 4],
                'text': f'Notification {i}',
                'hour_of_day': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'clicked': np.random.random() < 0.15,  # 15% CTR
                'unsubscribed': np.random.random() < 0.01  # 1% unsub rate
            })

    return pd.DataFrame(events)


@pytest.mark.slow
def test_full_pipeline_integration():
    """
    Complete end-to-end test of entire system.

    Steps:
    1. Generate synthetic data
    2. Build training dataset with features
    3. Train click and complaint models
    4. Initialize decision maker
    5. Run simulation
    6. Verify metrics and constraints
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Generate data
        print("\n1. Generating synthetic data...")
        raw_df = create_synthetic_e2e_dataset(n_users=30, events_per_user=30)
        raw_path = tmpdir / "raw_events.parquet"
        raw_df.to_parquet(raw_path)

        assert raw_path.exists()
        print(f"   ✓ Generated {len(raw_df)} events from {raw_df['user_id'].nunique()} users")

        # Step 2: Build training dataset
        print("\n2. Building training dataset with features...")
        output_path = tmpdir / "processed.parquet"
        train_df, test_df = build_training_dataset(
            raw_path,
            output_path,
            test_split=0.2
        )

        assert len(train_df) > 0
        assert len(test_df) > 0
        print(f"   ✓ Train: {len(train_df)}, Test: {len(test_df)}")

        # Verify features are present
        required_features = ['notif_count_24h', 'notif_count_168h', 'open_rate_168h']
        for feat in required_features:
            assert feat in train_df.columns, f"Missing feature: {feat}"

        # Step 3: Train models
        print("\n3. Training models...")

        click_model = BaseClickModel()
        click_metrics = click_model.train(
            train_df,
            target_col='clicked',
            valid_df=test_df,
            verbose=False
        )

        complaint_model = BaseComplaintModel()
        complaint_metrics = complaint_model.train(
            train_df,
            target_col='unsubscribed',
            valid_df=test_df,
            verbose=False
        )

        print(f"   ✓ Click model - Train AUC: {click_metrics.get('train_auc', 0):.3f}, Valid AUC: {click_metrics.get('valid_auc', 0):.3f}")
        print(f"   ✓ Complaint model - Train AUC: {complaint_metrics.get('train_auc', 0):.3f}, Valid AUC: {complaint_metrics.get('valid_auc', 0):.3f}")

        # Save models
        click_model_path = tmpdir / "click_model.pkl"
        complaint_model_path = tmpdir / "complaint_model.pkl"

        with open(click_model_path, 'wb') as f:
            pickle.dump(click_model, f)
        with open(complaint_model_path, 'wb') as f:
            pickle.dump(complaint_model, f)

        assert click_model_path.exists()
        assert complaint_model_path.exists()
        print(f"   ✓ Models saved to disk")

        # Step 4: Initialize decision maker
        print("\n4. Initializing decision maker...")

        fatigue_model = FatigueModel()
        llm_client = MockLLMClient()
        llm_client.load_model()
        scorer = NotificationScorer()

        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=fatigue_model,
            llm_client=llm_client,
            scorer=scorer
        )

        print(f"   ✓ Decision maker initialized")

        # Step 5: Run simulation
        print("\n5. Running offline simulation...")

        simulator = NotificationSimulator(decision_maker)

        # Simulate without LLM (faster)
        start_time = time.time()
        result = simulator.simulate(test_df, use_llm=False)
        sim_time = time.time() - start_time

        print(f"   ✓ Simulation complete in {sim_time:.2f}s")
        print(f"   • Processed {result.total_candidates} candidates")
        print(f"   • Sent {result.sent} ({result.sent/result.total_candidates:.1%})")
        print(f"   • True Positives: {result.true_positives}")
        print(f"   • Decision latency: {result.decision_latency_ms:.1f}ms")

        # Step 6: Compute and verify metrics
        print("\n6. Computing metrics...")

        metrics = MetricsCalculator.compute_metrics(result)

        print(f"   • Precision: {metrics.precision:.2%}")
        print(f"   • Recall: {metrics.recall:.2%}")
        print(f"   • F1 Score: {metrics.f1_score:.2%}")
        print(f"   • CTR: {metrics.click_through_rate:.2%}")
        print(f"   • Complaint Rate: {metrics.complaint_rate:.2%}")

        # Verify metrics are in valid ranges
        assert 0 <= metrics.precision <= 1, "Precision out of range"
        assert 0 <= metrics.recall <= 1, "Recall out of range"
        assert 0 <= metrics.f1_score <= 1, "F1 score out of range"
        assert 0 <= metrics.click_through_rate <= 1, "CTR out of range"
        assert 0 <= metrics.complaint_rate <= 1, "Complaint rate out of range"

        # Verify confusion matrix adds up
        assert result.true_positives + result.false_positives + result.true_negatives + result.false_negatives == result.total_candidates

        # Step 7: Check resource constraints
        print("\n7. Checking resource constraints...")
        import psutil
        import os
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"   • Memory usage: {mem_mb:.1f} MB")

        # Should be under 2GB constraint (being generous for test environment)
        assert mem_mb < 2048, f"Memory too high: {mem_mb} MB"

        # Latency should be under 500ms (per decision)
        avg_latency_per_decision = result.decision_latency_ms / result.total_candidates
        print(f"   • Avg latency per decision: {avg_latency_per_decision:.2f}ms")

        # Being generous - batch processing should be very fast
        assert avg_latency_per_decision < 100, f"Latency too high: {avg_latency_per_decision}ms"

        print("\n✓ End-to-end integration test PASSED")


@pytest.mark.slow
def test_pipeline_with_llm():
    """Test pipeline with LLM enabled for top-K candidates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate smaller dataset for faster testing
        raw_df = create_synthetic_e2e_dataset(n_users=20, events_per_user=20)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        # Build dataset
        output_path = tmpdir / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path, test_split=0.2)

        # Train models
        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        # Initialize with LLM
        fatigue_model = FatigueModel()
        llm_client = MockLLMClient()
        llm_client.load_model()

        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=fatigue_model,
            llm_client=llm_client
        )

        # Run simulation with LLM for top-3
        simulator = NotificationSimulator(decision_maker)
        result = simulator.simulate(test_df, use_llm=True, top_k_llm=3)

        # Verify results
        assert result.total_candidates > 0
        assert result.sent >= 0
        assert result.decision_latency_ms > 0

        print(f"\n✓ LLM-enabled simulation: {result.sent}/{result.total_candidates} sent")


@pytest.mark.slow
def test_model_persistence():
    """Test that models can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate and process data
        raw_df = create_synthetic_e2e_dataset(n_users=20, events_per_user=20)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        output_path = tmpdir / "proc.parquet"
        train_df, _ = build_training_dataset(raw_path, output_path, test_split=0.2)

        # Train model
        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        # Make predictions before saving
        predictions_before = click_model.predict(train_df.head(10))

        # Save model
        model_path = tmpdir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(click_model, f)

        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Make predictions after loading
        predictions_after = loaded_model.predict(train_df.head(10))

        # Predictions should be identical
        np.testing.assert_array_almost_equal(predictions_before, predictions_after)

        print("\n✓ Model persistence test PASSED")


@pytest.mark.slow
def test_resource_constraints_throughout_pipeline():
    """Test that resource constraints are met at every stage."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Baseline memory
        mem_baseline = process.memory_info().rss / 1024 / 1024

        # Create dataset
        raw_df = create_synthetic_e2e_dataset(n_users=30, events_per_user=30)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        mem1 = process.memory_info().rss / 1024 / 1024
        print(f"\nMemory after data creation: {mem1:.1f} MB (Δ{mem1 - mem_baseline:.1f} MB)")
        assert mem1 < 2048, f"Memory after data creation: {mem1} MB"

        # Build dataset
        output_path = tmpdir / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path, test_split=0.2)

        mem2 = process.memory_info().rss / 1024 / 1024
        print(f"Memory after dataset building: {mem2:.1f} MB (Δ{mem2 - mem1:.1f} MB)")
        assert mem2 < 2048, f"Memory after dataset building: {mem2} MB"

        # Train models
        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        mem3 = process.memory_info().rss / 1024 / 1024
        print(f"Memory after model training: {mem3:.1f} MB (Δ{mem3 - mem2:.1f} MB)")
        assert mem3 < 2048, f"Memory after model training: {mem3} MB"

        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        mem4 = process.memory_info().rss / 1024 / 1024
        print(f"Memory after both models: {mem4:.1f} MB (Δ{mem4 - mem3:.1f} MB)")
        assert mem4 < 2048, f"Memory after both models: {mem4} MB"

        # Run simulation
        fatigue_model = FatigueModel()
        llm_client = MockLLMClient()
        llm_client.load_model()

        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=fatigue_model,
            llm_client=llm_client
        )

        simulator = NotificationSimulator(decision_maker)
        result = simulator.simulate(test_df, use_llm=False)

        mem5 = process.memory_info().rss / 1024 / 1024
        print(f"Memory after simulation: {mem5:.1f} MB (Δ{mem5 - mem4:.1f} MB)")
        assert mem5 < 2048, f"Memory after simulation: {mem5} MB"

        print(f"\n✓ All stages under {DEFAULT_CONFIG.resources.max_memory_mb} MB constraint")


@pytest.mark.slow
def test_threshold_sweep_integration():
    """Test threshold sweep functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data
        raw_df = create_synthetic_e2e_dataset(n_users=20, events_per_user=20)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        # Build and train
        output_path = tmpdir / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path, test_split=0.2)

        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        # Initialize decision maker
        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=FatigueModel(),
            llm_client=MockLLMClient()
        )

        # Run threshold sweep
        simulator = NotificationSimulator(decision_maker)
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        results_df = simulator.simulate_with_threshold_sweep(test_df, thresholds)

        # Verify results
        assert len(results_df) == len(thresholds)
        assert 'threshold' in results_df.columns
        assert 'sent' in results_df.columns
        assert 'precision' in results_df.columns
        assert 'recall' in results_df.columns
        assert 'f1' in results_df.columns

        # As threshold increases, sent should generally decrease
        assert results_df.iloc[0]['sent'] >= results_df.iloc[-1]['sent']

        print(f"\n✓ Threshold sweep test PASSED")
        print(f"\nThreshold sweep results:")
        print(results_df.to_string())


@pytest.mark.slow
def test_api_integration():
    """Test that API works with trained models."""
    from fastapi.testclient import TestClient
    from src.api import server
    from src.api.server import app

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and train models
        raw_df = create_synthetic_e2e_dataset(n_users=20, events_per_user=20)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        output_path = tmpdir / "proc.parquet"
        train_df, _ = build_training_dataset(raw_path, output_path, test_split=0.2)

        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        # Inject into API server
        server.click_model = click_model
        server.complaint_model = complaint_model
        server.fatigue_model = FatigueModel()
        server.llm_client = MockLLMClient()

        server.decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=server.fatigue_model,
            llm_client=server.llm_client
        )

        # Test API
        client = TestClient(app)

        request_data = {
            "user_features": {
                "user_id": 123,
                "notif_count_24h": 2,
                "notif_count_7d": 10,
                "notif_count_30d": 30,
                "open_rate_7d": 0.15,
                "open_rate_30d": 0.12
            },
            "candidate": {
                "template_id": "test_001",
                "channel": "push",
                "notification_type": "promo",
                "text": "Test notification",
                "hour_of_day": 14,
                "day_of_week": 2
            }
        }

        response = client.post("/decide", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "should_send" in data
        assert "final_score" in data
        assert "p_click" in data
        assert "p_complaint" in data

        print(f"\n✓ API integration test PASSED")
        print(f"  Decision: {'SEND' if data['should_send'] else 'REJECT'}")
        print(f"  Score: {data['final_score']:.3f}")
