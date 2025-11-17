"""
Evaluation metrics for notification system.
Provides standard metrics for assessing policy performance.
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.simulation.simulator import SimulationResult


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    # Classification metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Business metrics
    send_rate: float
    click_through_rate: float
    complaint_rate: float

    # Volume metrics
    total_sent: int
    total_clicks: int
    total_complaints: int

    # Efficiency metrics
    clicks_per_send: float
    complaints_per_send: float


class MetricsCalculator:
    """
    Calculates evaluation metrics from simulation results.

    Provides standard ML metrics plus business-specific metrics.
    """

    @staticmethod
    def compute_metrics(result: SimulationResult) -> MetricsSummary:
        """
        Compute comprehensive metrics from simulation result.

        Args:
            result: SimulationResult object

        Returns:
            MetricsSummary with all metrics
        """
        # Classification metrics
        precision = (
            result.true_positives / result.sent
            if result.sent > 0 else 0.0
        )

        recall = (
            result.true_positives / result.actual_clicks
            if result.actual_clicks > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        accuracy = (
            (result.true_positives + result.true_negatives) / result.total_candidates
            if result.total_candidates > 0 else 0.0
        )

        # Business metrics
        send_rate = result.sent / result.total_candidates if result.total_candidates > 0 else 0.0

        click_through_rate = (
            result.true_positives / result.sent
            if result.sent > 0 else 0.0
        )

        # Complaints among sent notifications
        complaints_sent = result.actual_complaints - result.complaints_avoided
        complaint_rate = (
            complaints_sent / result.sent
            if result.sent > 0 else 0.0
        )

        # Efficiency
        clicks_per_send = result.true_positives / result.sent if result.sent > 0 else 0.0
        complaints_per_send = complaints_sent / result.sent if result.sent > 0 else 0.0

        return MetricsSummary(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            send_rate=send_rate,
            click_through_rate=click_through_rate,
            complaint_rate=complaint_rate,
            total_sent=result.sent,
            total_clicks=result.true_positives,
            total_complaints=complaints_sent,
            clicks_per_send=clicks_per_send,
            complaints_per_send=complaints_per_send
        )

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        """Compute precision: TP / (TP + FP)."""
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        """Compute recall: TP / (TP + FN)."""
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Compute F1 score: 2 * (precision * recall) / (precision + recall)."""
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def compute_lift(
        treatment_ctr: float,
        control_ctr: float
    ) -> float:
        """
        Compute lift of treatment over control.

        Args:
            treatment_ctr: Click-through rate in treatment group
            control_ctr: Click-through rate in control group

        Returns:
            Lift (relative improvement)
        """
        return (treatment_ctr - control_ctr) / control_ctr if control_ctr > 0 else 0.0

    @staticmethod
    def compute_incremental_value(
        policy_result: SimulationResult,
        baseline_result: SimulationResult
    ) -> Dict[str, float]:
        """
        Compute incremental value of policy over baseline.

        Args:
            policy_result: Results from policy being evaluated
            baseline_result: Results from baseline policy

        Returns:
            Dict with incremental metrics
        """
        # Additional clicks captured
        incremental_clicks = policy_result.true_positives - baseline_result.true_positives

        # Reduction in notifications sent
        volume_reduction = baseline_result.sent - policy_result.sent

        # Complaints avoided
        policy_complaints = policy_result.actual_complaints - policy_result.complaints_avoided
        baseline_complaints = baseline_result.actual_complaints - baseline_result.complaints_avoided
        incremental_complaint_reduction = baseline_complaints - policy_complaints

        return {
            'incremental_clicks': incremental_clicks,
            'volume_reduction': volume_reduction,
            'incremental_complaint_reduction': incremental_complaint_reduction,
            'efficiency_gain': (
                policy_result.true_positives / policy_result.sent -
                baseline_result.true_positives / baseline_result.sent
            ) if policy_result.sent > 0 and baseline_result.sent > 0 else 0.0
        }


class ABTestMetrics:
    """
    Metrics for A/B testing notification policies.

    Compares two policies statistically.
    """

    @staticmethod
    def compare_policies(
        treatment_result: SimulationResult,
        control_result: SimulationResult
    ) -> Dict[str, any]:
        """
        Compare treatment vs control policies.

        Args:
            treatment_result: Results from treatment policy
            control_result: Results from control policy

        Returns:
            Dict with comparison metrics
        """
        # Click-through rates
        treatment_ctr = (
            treatment_result.true_positives / treatment_result.sent
            if treatment_result.sent > 0 else 0.0
        )

        control_ctr = (
            control_result.true_positives / control_result.sent
            if control_result.sent > 0 else 0.0
        )

        # Lift
        lift = MetricsCalculator.compute_lift(treatment_ctr, control_ctr)

        # Complaint rates
        treatment_complaints = treatment_result.actual_complaints - treatment_result.complaints_avoided
        control_complaints = control_result.actual_complaints - control_result.complaints_avoided

        treatment_complaint_rate = (
            treatment_complaints / treatment_result.sent
            if treatment_result.sent > 0 else 0.0
        )

        control_complaint_rate = (
            control_complaints / control_result.sent
            if control_result.sent > 0 else 0.0
        )

        # Statistical significance (simple z-test)
        z_score, p_value = ABTestMetrics._compute_significance(
            treatment_result.true_positives,
            treatment_result.sent,
            control_result.true_positives,
            control_result.sent
        )

        return {
            'treatment': {
                'ctr': treatment_ctr,
                'sent': treatment_result.sent,
                'clicks': treatment_result.true_positives,
                'complaint_rate': treatment_complaint_rate
            },
            'control': {
                'ctr': control_ctr,
                'sent': control_result.sent,
                'clicks': control_result.true_positives,
                'complaint_rate': control_complaint_rate
            },
            'comparison': {
                'lift': lift,
                'absolute_ctr_diff': treatment_ctr - control_ctr,
                'relative_ctr_improvement': lift,
                'complaint_rate_diff': treatment_complaint_rate - control_complaint_rate,
                'z_score': z_score,
                'p_value': p_value,
                'significant_at_95': p_value < 0.05
            }
        }

    @staticmethod
    def _compute_significance(
        treatment_successes: int,
        treatment_total: int,
        control_successes: int,
        control_total: int
    ) -> tuple:
        """
        Compute statistical significance using z-test for proportions.

        Args:
            treatment_successes: Number of clicks in treatment
            treatment_total: Total sent in treatment
            control_successes: Number of clicks in control
            control_total: Total sent in control

        Returns:
            Tuple of (z_score, p_value)
        """
        if treatment_total == 0 or control_total == 0:
            return 0.0, 1.0

        # Sample proportions
        p1 = treatment_successes / treatment_total
        p2 = control_successes / control_total

        # Pooled proportion
        p_pool = (treatment_successes + control_successes) / (treatment_total + control_total)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/treatment_total + 1/control_total))

        if se == 0:
            return 0.0, 1.0

        # Z-score
        z = (p1 - p2) / se

        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return float(z), float(p_value)


def format_metrics_report(metrics: MetricsSummary) -> str:
    """
    Format metrics as human-readable report.

    Args:
        metrics: MetricsSummary object

    Returns:
        Formatted string report
    """
    report = f"""
Notification System Performance Metrics
========================================

Classification Metrics:
  Precision:  {metrics.precision:.2%}
  Recall:     {metrics.recall:.2%}
  F1 Score:   {metrics.f1_score:.2%}
  Accuracy:   {metrics.accuracy:.2%}

Business Metrics:
  Send Rate:          {metrics.send_rate:.2%}
  Click-Through Rate: {metrics.click_through_rate:.2%}
  Complaint Rate:     {metrics.complaint_rate:.2%}

Volume:
  Total Sent:       {metrics.total_sent:,}
  Total Clicks:     {metrics.total_clicks:,}
  Total Complaints: {metrics.total_complaints:,}

Efficiency:
  Clicks per Send:      {metrics.clicks_per_send:.3f}
  Complaints per Send:  {metrics.complaints_per_send:.3f}
"""
    return report


def compute_threshold_metrics(
    threshold_results: List[Dict]
) -> Dict[str, float]:
    """
    Find optimal threshold based on various criteria.

    Args:
        threshold_results: List of dicts with threshold metrics

    Returns:
        Dict with optimal thresholds for different objectives
    """
    import pandas as pd

    df = pd.DataFrame(threshold_results)

    optimal = {}

    # Best F1
    if 'f1' in df.columns:
        best_f1_idx = df['f1'].idxmax()
        optimal['best_f1_threshold'] = df.loc[best_f1_idx, 'threshold']
        optimal['best_f1_score'] = df.loc[best_f1_idx, 'f1']

    # Best precision
    if 'precision' in df.columns:
        best_prec_idx = df['precision'].idxmax()
        optimal['best_precision_threshold'] = df.loc[best_prec_idx, 'threshold']
        optimal['best_precision_score'] = df.loc[best_prec_idx, 'precision']

    # Best recall
    if 'recall' in df.columns:
        best_recall_idx = df['recall'].idxmax()
        optimal['best_recall_threshold'] = df.loc[best_recall_idx, 'threshold']
        optimal['best_recall_score'] = df.loc[best_recall_idx, 'recall']

    return optimal
