"""Tests for simulation and metrics."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.simulation.simulator import NotificationSimulator, SimulationResult
from src.simulation.metrics import (
    MetricsCalculator,
    ABTestMetrics,
    MetricsSummary,
    format_metrics_report,
    compute_threshold_metrics
)
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.decision import NotificationDecisionMaker
from src.config import DEFAULT_CONFIG

# Import test utilities
import sys
sys.path.insert(0, 'tests')
from test_base_model import create_synthetic_dataset


def create_test_decision_maker():
    """Create a decision maker for testing."""
    df = create_synthetic_dataset(n_samples=500)

    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)

    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    fatigue_model = FatigueModel()
    llm_client = MockLLMClient()

    return NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=fatigue_model,
        llm_client=llm_client
    )


# ===== Simulator Tests =====

def test_simulator_initialization():
    """Test simulator initialization."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    assert simulator.decision_maker is not None
    assert simulator.config is not None


def test_simulate_basic():
    """Test basic simulation."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    # Create test events
    df = create_synthetic_dataset(n_samples=100)

    # Run simulation
    result = simulator.simulate(df, use_llm=False)

    assert isinstance(result, SimulationResult)
    assert result.total_candidates == 100
    assert result.sent + result.not_sent == 100
    assert result.true_positives >= 0
    assert result.false_positives >= 0
    assert result.decision_latency_ms >= 0


def test_simulate_with_llm():
    """Test simulation with LLM."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    df = create_synthetic_dataset(n_samples=50)

    result = simulator.simulate(df, use_llm=True, top_k_llm=5)

    assert isinstance(result, SimulationResult)
    assert result.total_candidates == 50


def test_simulation_confusion_matrix():
    """Test that confusion matrix values are consistent."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    df = create_synthetic_dataset(n_samples=100)
    result = simulator.simulate(df, use_llm=False)

    # TP + FP + TN + FN should equal total
    total = result.true_positives + result.false_positives + result.true_negatives + result.false_negatives
    assert total == result.total_candidates

    # TP + FP should equal sent
    assert result.true_positives + result.false_positives == result.sent

    # TN + FN should equal not_sent
    assert result.true_negatives + result.false_negatives == result.not_sent


def test_compare_policies():
    """Test policy comparison."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    df = create_synthetic_dataset(n_samples=100)

    # Compare against send-all baseline
    comparison = simulator.compare_policies(df, baseline_send_all=True)

    assert 'current_policy' in comparison
    assert 'baseline' in comparison
    assert 'improvement' in comparison

    # Baseline should send all
    assert comparison['baseline']['sent'] == 100

    # Current policy should send fewer
    assert comparison['current_policy']['sent'] <= 100

    # Should have volume reduction
    assert comparison['improvement']['volume_reduction'] >= 0


def test_threshold_sweep():
    """Test threshold sweep."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    df = create_synthetic_dataset(n_samples=100)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    results_df = simulator.simulate_with_threshold_sweep(df, thresholds)

    assert len(results_df) == len(thresholds)
    assert 'threshold' in results_df.columns
    assert 'precision' in results_df.columns
    assert 'recall' in results_df.columns
    assert 'f1' in results_df.columns

    # Higher thresholds should generally send fewer
    sent_counts = results_df['sent'].values
    # Not strictly monotonic due to discrete nature, but trend should hold
    assert sent_counts[0] >= sent_counts[-1]


# ===== Metrics Calculator Tests =====

def test_metrics_calculator_basic():
    """Test basic metrics calculation."""
    # Create a simple simulation result
    result = SimulationResult(
        total_candidates=100,
        sent=60,
        not_sent=40,
        true_positives=30,  # 30 sent and clicked
        false_positives=30,  # 30 sent but not clicked
        true_negatives=35,  # 35 not sent and wouldn't click
        false_negatives=5,  # 5 not sent but would have clicked
        actual_clicks=35,  # 30 + 5
        actual_complaints=5,
        complaints_avoided=2,
        mean_score_sent=0.5,
        mean_score_rejected=0.2,
        decision_latency_ms=100.0
    )

    metrics = MetricsCalculator.compute_metrics(result)

    assert isinstance(metrics, MetricsSummary)

    # Check precision = TP / (TP + FP) = 30 / 60 = 0.5
    assert abs(metrics.precision - 0.5) < 0.001

    # Check recall = TP / (TP + FN) = 30 / 35 = 0.857
    assert abs(metrics.recall - 30/35) < 0.001

    # Check F1
    expected_f1 = 2 * 0.5 * (30/35) / (0.5 + 30/35)
    assert abs(metrics.f1_score - expected_f1) < 0.001


def test_precision_computation():
    """Test precision computation."""
    precision = MetricsCalculator.precision(tp=30, fp=20)
    assert abs(precision - 0.6) < 0.001

    # Edge case: no predictions
    precision_zero = MetricsCalculator.precision(tp=0, fp=0)
    assert precision_zero == 0.0


def test_recall_computation():
    """Test recall computation."""
    recall = MetricsCalculator.recall(tp=30, fn=10)
    assert abs(recall - 0.75) < 0.001

    # Edge case
    recall_zero = MetricsCalculator.recall(tp=0, fn=0)
    assert recall_zero == 0.0


def test_f1_score_computation():
    """Test F1 score computation."""
    f1 = MetricsCalculator.f1_score(precision=0.8, recall=0.6)
    expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
    assert abs(f1 - expected) < 0.001

    # Edge case
    f1_zero = MetricsCalculator.f1_score(precision=0.0, recall=0.0)
    assert f1_zero == 0.0


def test_lift_computation():
    """Test lift computation."""
    lift = MetricsCalculator.compute_lift(treatment_ctr=0.15, control_ctr=0.10)
    assert abs(lift - 0.5) < 0.001  # 50% improvement

    # Edge case
    lift_zero = MetricsCalculator.compute_lift(treatment_ctr=0.1, control_ctr=0.0)
    assert lift_zero == 0.0


def test_incremental_value():
    """Test incremental value computation."""
    policy_result = SimulationResult(
        total_candidates=100,
        sent=50,
        not_sent=50,
        true_positives=25,
        false_positives=25,
        true_negatives=45,
        false_negatives=5,
        actual_clicks=30,
        actual_complaints=3,
        complaints_avoided=1,
        mean_score_sent=0.5,
        mean_score_rejected=0.2,
        decision_latency_ms=100.0
    )

    baseline_result = SimulationResult(
        total_candidates=100,
        sent=100,
        not_sent=0,
        true_positives=30,
        false_positives=70,
        true_negatives=0,
        false_negatives=0,
        actual_clicks=30,
        actual_complaints=5,
        complaints_avoided=0,
        mean_score_sent=0.3,
        mean_score_rejected=0.0,
        decision_latency_ms=50.0
    )

    incremental = MetricsCalculator.compute_incremental_value(policy_result, baseline_result)

    assert 'incremental_clicks' in incremental
    assert 'volume_reduction' in incremental
    assert 'incremental_complaint_reduction' in incremental

    # Volume reduction = 100 - 50 = 50
    assert incremental['volume_reduction'] == 50


# ===== A/B Test Metrics Tests =====

def test_ab_test_comparison():
    """Test A/B test comparison."""
    treatment_result = SimulationResult(
        total_candidates=100,
        sent=60,
        not_sent=40,
        true_positives=30,
        false_positives=30,
        true_negatives=35,
        false_negatives=5,
        actual_clicks=35,
        actual_complaints=3,
        complaints_avoided=1,
        mean_score_sent=0.5,
        mean_score_rejected=0.2,
        decision_latency_ms=100.0
    )

    control_result = SimulationResult(
        total_candidates=100,
        sent=60,
        not_sent=40,
        true_positives=24,  # Lower CTR
        false_positives=36,
        true_negatives=35,
        false_negatives=5,
        actual_clicks=29,
        actual_complaints=4,
        complaints_avoided=1,
        mean_score_sent=0.45,
        mean_score_rejected=0.2,
        decision_latency_ms=100.0
    )

    comparison = ABTestMetrics.compare_policies(treatment_result, control_result)

    assert 'treatment' in comparison
    assert 'control' in comparison
    assert 'comparison' in comparison

    # Treatment should have higher CTR
    assert comparison['treatment']['ctr'] > comparison['control']['ctr']

    # Should have positive lift
    assert comparison['comparison']['lift'] > 0


def test_statistical_significance():
    """Test statistical significance computation."""
    # Large sample with clear difference should be significant
    z, p = ABTestMetrics._compute_significance(
        treatment_successes=150,
        treatment_total=1000,
        control_successes=100,
        control_total=1000
    )

    assert p < 0.05  # Should be significant

    # Small difference might not be significant
    z2, p2 = ABTestMetrics._compute_significance(
        treatment_successes=51,
        treatment_total=100,
        control_successes=49,
        control_total=100
    )

    # Likely not significant with such small difference
    # (actual p-value depends on sample size)


# ===== Metrics Formatting Tests =====

def test_format_metrics_report():
    """Test metrics report formatting."""
    metrics = MetricsSummary(
        precision=0.75,
        recall=0.80,
        f1_score=0.77,
        accuracy=0.85,
        send_rate=0.60,
        click_through_rate=0.25,
        complaint_rate=0.02,
        total_sent=600,
        total_clicks=150,
        total_complaints=12,
        clicks_per_send=0.25,
        complaints_per_send=0.02
    )

    report = format_metrics_report(metrics)

    assert isinstance(report, str)
    assert 'Precision' in report
    assert '75.00%' in report or '0.75' in report
    assert 'Recall' in report
    assert 'F1 Score' in report


def test_compute_threshold_metrics():
    """Test threshold metrics computation."""
    results = [
        {'threshold': 0.2, 'precision': 0.5, 'recall': 0.9, 'f1': 0.64},
        {'threshold': 0.4, 'precision': 0.7, 'recall': 0.7, 'f1': 0.7},
        {'threshold': 0.6, 'precision': 0.9, 'recall': 0.5, 'f1': 0.64},
    ]

    optimal = compute_threshold_metrics(results)

    assert 'best_f1_threshold' in optimal
    assert 'best_precision_threshold' in optimal
    assert 'best_recall_threshold' in optimal

    # Best F1 should be at 0.4
    assert optimal['best_f1_threshold'] == 0.4

    # Best precision at 0.6
    assert optimal['best_precision_threshold'] == 0.6

    # Best recall at 0.2
    assert optimal['best_recall_threshold'] == 0.2


# ===== Integration Tests =====

def test_end_to_end_simulation():
    """Test complete simulation pipeline."""
    # Create decision maker
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    # Create test data
    df = create_synthetic_dataset(n_samples=200)

    # Run simulation
    result = simulator.simulate(df, use_llm=False)

    # Compute metrics
    metrics = MetricsCalculator.compute_metrics(result)

    # Validate
    assert metrics.precision >= 0 and metrics.precision <= 1
    assert metrics.recall >= 0 and metrics.recall <= 1
    assert metrics.f1_score >= 0 and metrics.f1_score <= 1
    assert metrics.send_rate >= 0 and metrics.send_rate <= 1

    # Generate report
    report = format_metrics_report(metrics)
    assert len(report) > 100


def test_simulation_with_different_data_sizes():
    """Test simulation scales with data size."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    for n in [50, 100, 200]:
        df = create_synthetic_dataset(n_samples=n)
        result = simulator.simulate(df, use_llm=False)

        assert result.total_candidates == n
        assert result.sent + result.not_sent == n


def test_simulation_reproducibility():
    """Test that simulation is reproducible."""
    dm = create_test_decision_maker()
    simulator = NotificationSimulator(dm)

    df = create_synthetic_dataset(n_samples=100)

    # Run twice
    result1 = simulator.simulate(df, use_llm=False)
    result2 = simulator.simulate(df, use_llm=False)

    # Results should be identical (deterministic mock LLM)
    assert result1.sent == result2.sent
    assert result1.true_positives == result2.true_positives


def test_metrics_with_edge_cases():
    """Test metrics handle edge cases gracefully."""
    # All sent, all clicked
    result_perfect = SimulationResult(
        total_candidates=100,
        sent=100,
        not_sent=0,
        true_positives=100,
        false_positives=0,
        true_negatives=0,
        false_negatives=0,
        actual_clicks=100,
        actual_complaints=0,
        complaints_avoided=0,
        mean_score_sent=0.9,
        mean_score_rejected=0.0,
        decision_latency_ms=100.0
    )

    metrics_perfect = MetricsCalculator.compute_metrics(result_perfect)
    assert metrics_perfect.precision == 1.0
    assert metrics_perfect.recall == 1.0

    # None sent
    result_none = SimulationResult(
        total_candidates=100,
        sent=0,
        not_sent=100,
        true_positives=0,
        false_positives=0,
        true_negatives=90,
        false_negatives=10,
        actual_clicks=10,
        actual_complaints=0,
        complaints_avoided=0,
        mean_score_sent=0.0,
        mean_score_rejected=0.3,
        decision_latency_ms=50.0
    )

    metrics_none = MetricsCalculator.compute_metrics(result_none)
    assert metrics_none.precision == 0.0
    assert metrics_none.recall == 0.0
