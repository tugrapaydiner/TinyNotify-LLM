"""
Offline simulator for notification system.
Replays historical events to evaluate policy performance.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import CandidateNotification, UserFeatures, Channel, NotificationType
from src.policy.decision import NotificationDecisionMaker

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    total_candidates: int
    sent: int
    not_sent: int
    true_positives: int  # Sent and clicked
    false_positives: int  # Sent but not clicked
    true_negatives: int  # Not sent and wouldn't have clicked
    false_negatives: int  # Not sent but would have clicked
    actual_clicks: int  # Total actual clicks in data
    actual_complaints: int  # Total actual complaints
    complaints_avoided: int  # Complaints we avoided by not sending
    mean_score_sent: float
    mean_score_rejected: float
    decision_latency_ms: float
    all_scores: Optional[List[Dict]] = None  # All scores for diagnostic purposes


class NotificationSimulator:
    """
    Simulates notification system on historical data.

    Evaluates policy performance by:
    1. Loading historical events
    2. Making decisions with current policy
    3. Comparing to actual outcomes
    4. Computing metrics
    """

    def __init__(
        self,
        decision_maker: NotificationDecisionMaker,
        config: Optional[Config] = None
    ):
        """
        Initialize simulator.

        Args:
            decision_maker: Configured decision maker
            config: Configuration object
        """
        self.decision_maker = decision_maker
        self.config = config if config is not None else DEFAULT_CONFIG

    def simulate(
        self,
        events_df: pd.DataFrame,
        use_llm: bool = False,
        top_k_llm: Optional[int] = None
    ) -> SimulationResult:
        """
        Run simulation on historical events.

        Args:
            events_df: DataFrame with historical notification events
            use_llm: Whether to use LLM in simulation
            top_k_llm: Number of top candidates to rate with LLM

        Returns:
            SimulationResult with metrics
        """
        import time

        logger.info(f"Starting simulation with {len(events_df)} events")

        # Convert events to candidates
        candidates = self._events_to_candidates(events_df)

        # Make decisions
        start_time = time.time()

        if use_llm:
            decisions = self.decision_maker.decide_batch(
                candidates,
                use_llm_for_top_k=True,
                top_k=top_k_llm or self.config.policy.top_k_for_llm
            )
        else:
            decisions = self.decision_maker.decide_batch(
                candidates,
                use_llm_for_top_k=False
            )

        latency_ms = (time.time() - start_time) * 1000

        # Compute metrics by comparing decisions to actual outcomes
        result = self._compute_metrics(events_df, decisions, latency_ms)

        logger.info(f"Simulation complete: {result.sent}/{result.total_candidates} sent")

        return result

    def _events_to_candidates(
        self,
        events_df: pd.DataFrame
    ) -> List[CandidateNotification]:
        """
        Convert event DataFrame rows to CandidateNotification objects.

        Args:
            events_df: DataFrame with events (must have user features attached)

        Returns:
            List of candidates
        """
        candidates = []

        for idx, row in events_df.iterrows():
            # Build UserFeatures from row
            user_features = UserFeatures(
                user_id=int(row['user_id']),
                timestamp=row['timestamp_send'],
                notif_count_24h=int(row.get('notif_count_24h', 0)),
                notif_count_7d=int(row.get('notif_count_168h', 0)),
                notif_count_30d=int(row.get('notif_count_720h', 0)),
                click_count_7d=int(row.get('click_count_168h', 0)),
                click_count_30d=int(row.get('click_count_720h', 0)),
                open_rate_7d=float(row.get('open_rate_168h', 0.0)),
                open_rate_30d=float(row.get('open_rate_720h', 0.0)),
                hours_since_last_notification=row.get('hours_since_last_notif_168h'),
                hours_since_last_click=row.get('hours_since_last_click_168h'),
                most_active_hour_bucket=row.get('hour_bucket'),
                complaint_count_30d=int(row.get('unsub_count_720h', 0))
            )

            # Parse channel and notification type
            channel_str = row.get('channel', 'push')
            notif_type_str = row.get('notification_type', 'product_tip')

            try:
                channel = Channel(channel_str) if isinstance(channel_str, str) else channel_str
            except:
                channel = Channel.PUSH

            try:
                notif_type = NotificationType(notif_type_str) if isinstance(notif_type_str, str) else notif_type_str
            except:
                notif_type = NotificationType.PRODUCT_TIP

            # Create candidate
            candidate = CandidateNotification(
                user_id=int(row['user_id']),
                template_id=row.get('template_id', 'unknown'),
                channel=channel,
                notification_type=notif_type,
                text=row.get('text', 'Notification'),
                timestamp=row['timestamp_send'],
                hour_of_day=int(row.get('hour_of_day', 12)),
                day_of_week=int(row.get('day_of_week', 0)),
                user_features=user_features
            )

            candidates.append(candidate)

        return candidates

    def _compute_metrics(
        self,
        events_df: pd.DataFrame,
        decisions: List,
        latency_ms: float
    ) -> SimulationResult:
        """
        Compute simulation metrics.

        Args:
            events_df: Original events with actual outcomes
            decisions: List of Decision objects
            latency_ms: Decision latency in milliseconds

        Returns:
            SimulationResult
        """
        # Extract actual outcomes
        actual_clicked = events_df['clicked'].values
        actual_complained = events_df.get('unsubscribed', pd.Series([False] * len(events_df))).values

        # Extract decisions
        predicted_send = np.array([d.should_send for d in decisions])

        # Compute confusion matrix for clicks
        sent_and_clicked = predicted_send & actual_clicked  # TP
        sent_and_not_clicked = predicted_send & ~actual_clicked  # FP
        not_sent_and_not_clicked = ~predicted_send & ~actual_clicked  # TN
        not_sent_and_clicked = ~predicted_send & actual_clicked  # FN

        tp = sent_and_clicked.sum()
        fp = sent_and_not_clicked.sum()
        tn = not_sent_and_not_clicked.sum()
        fn = not_sent_and_clicked.sum()

        # Complaints
        total_complaints = actual_complained.sum()

        # Complaints avoided (wouldn't have sent to users who complained)
        complaints_avoided = (~predicted_send & actual_complained).sum()

        # Scores
        sent_scores = [d.scored.final_score for d in decisions if d.should_send]
        rejected_scores = [d.scored.final_score for d in decisions if not d.should_send]

        mean_score_sent = float(np.mean(sent_scores)) if sent_scores else 0.0
        mean_score_rejected = float(np.mean(rejected_scores)) if rejected_scores else 0.0

        # Collect all scores for diagnostics
        all_scores = []
        for d in decisions:
            all_scores.append({
                'p_click': d.scored.p_click,
                'p_complaint': d.scored.p_complaint,
                'base_score': d.scored.base_score,
                'final_score': d.scored.final_score,
                'decision': 'SEND' if d.should_send else 'REJECT'
            })

        return SimulationResult(
            total_candidates=len(decisions),
            sent=predicted_send.sum(),
            not_sent=(~predicted_send).sum(),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            actual_clicks=int(actual_clicked.sum()),
            actual_complaints=int(total_complaints),
            complaints_avoided=int(complaints_avoided),
            mean_score_sent=mean_score_sent,
            mean_score_rejected=mean_score_rejected,
            decision_latency_ms=latency_ms,
            all_scores=all_scores
        )

    def compare_policies(
        self,
        events_df: pd.DataFrame,
        baseline_send_all: bool = True
    ) -> Dict[str, any]:
        """
        Compare current policy against baseline.

        Args:
            events_df: Historical events
            baseline_send_all: If True, baseline sends all; else sends none

        Returns:
            Dict with comparison metrics
        """
        # Simulate with current policy
        current_result = self.simulate(events_df, use_llm=False)

        # Baseline metrics
        actual_clicks = events_df['clicked'].sum()
        actual_complaints = events_df.get('unsubscribed', pd.Series([False] * len(events_df))).sum()

        if baseline_send_all:
            baseline_sent = len(events_df)
            baseline_clicks_captured = actual_clicks
            baseline_complaints = actual_complaints
        else:
            baseline_sent = 0
            baseline_clicks_captured = 0
            baseline_complaints = 0

        # Current policy performance
        current_clicks_captured = current_result.true_positives
        current_complaints = current_result.sent - current_result.complaints_avoided

        # Compute relative improvements
        click_capture_rate_current = (
            current_clicks_captured / actual_clicks if actual_clicks > 0 else 0
        )

        click_capture_rate_baseline = (
            baseline_clicks_captured / actual_clicks if actual_clicks > 0 else 0
        )

        volume_reduction = (
            (baseline_sent - current_result.sent) / baseline_sent
            if baseline_sent > 0 else 0
        )

        return {
            'current_policy': {
                'sent': current_result.sent,
                'clicks_captured': current_clicks_captured,
                'click_capture_rate': click_capture_rate_current,
                'complaints': current_complaints,
                'precision': current_result.true_positives / current_result.sent if current_result.sent > 0 else 0
            },
            'baseline': {
                'sent': baseline_sent,
                'clicks_captured': baseline_clicks_captured,
                'click_capture_rate': click_capture_rate_baseline,
                'complaints': baseline_complaints
            },
            'improvement': {
                'volume_reduction': volume_reduction,
                'complaint_reduction': (baseline_complaints - current_complaints) / baseline_complaints if baseline_complaints > 0 else 0,
                'clicks_lost': baseline_clicks_captured - current_clicks_captured,
                'clicks_lost_pct': (baseline_clicks_captured - current_clicks_captured) / baseline_clicks_captured if baseline_clicks_captured > 0 else 0
            }
        }

    def simulate_with_threshold_sweep(
        self,
        events_df: pd.DataFrame,
        thresholds: List[float]
    ) -> pd.DataFrame:
        """
        Simulate with different score thresholds.

        Args:
            events_df: Historical events
            thresholds: List of thresholds to try

        Returns:
            DataFrame with metrics for each threshold
        """
        results = []

        original_threshold = self.decision_maker.config.policy.score_threshold

        for threshold in thresholds:
            # Temporarily update threshold
            self.decision_maker.config.policy.score_threshold = threshold

            # Run simulation
            result = self.simulate(events_df, use_llm=False)

            # Compute derived metrics
            precision = result.true_positives / result.sent if result.sent > 0 else 0
            recall = result.true_positives / result.actual_clicks if result.actual_clicks > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'threshold': threshold,
                'sent': result.sent,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'complaints_avoided': result.complaints_avoided
            })

        # Restore original threshold
        self.decision_maker.config.policy.score_threshold = original_threshold

        return pd.DataFrame(results)
