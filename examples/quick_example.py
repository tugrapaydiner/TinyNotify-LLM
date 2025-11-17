"""
Quick example showing how to use TinyNotify-LLM programmatically.

This example demonstrates:
1. Creating synthetic data
2. Training models
3. Making decisions
4. Running simulations
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile

from src.features.build_dataset import build_training_dataset
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.decision import NotificationDecisionMaker
from src.features.schema import UserFeatures, CandidateNotification, Channel, NotificationType
from src.simulation.simulator import NotificationSimulator
from src.simulation.metrics import MetricsCalculator


def create_sample_data(n_users=50, events_per_user=30):
    """Create sample notification event data."""
    print("📊 Creating sample data...")

    events = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for user_id in range(n_users):
        # User behavior pattern
        engagement_level = np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.5, 0.3])

        if engagement_level == 'high':
            click_prob = 0.25
            unsub_prob = 0.005
        elif engagement_level == 'medium':
            click_prob = 0.15
            unsub_prob = 0.01
        else:
            click_prob = 0.05
            unsub_prob = 0.02

        for i in range(events_per_user):
            timestamp = base_time + timedelta(hours=user_id * 10 + i * 2)

            events.append({
                'user_id': user_id,
                'template_id': f'template_{i % 10}',
                'timestamp_send': timestamp,
                'channel': np.random.choice(['push', 'email', 'sms', 'in_app']),
                'notification_type': np.random.choice(['promo', 'product_tip', 'important', 'system_alert']),
                'text': f'Notification {i}',
                'hour_of_day': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'clicked': np.random.random() < click_prob,
                'unsubscribed': np.random.random() < unsub_prob
            })

    df = pd.DataFrame(events)
    print(f"   ✓ Created {len(df)} events from {df['user_id'].nunique()} users")
    print(f"   ✓ Click rate: {df['clicked'].mean():.1%}")
    print(f"   ✓ Unsubscribe rate: {df['unsubscribed'].mean():.1%}")

    return df


def main():
    print("\n" + "="*60)
    print("TinyNotify-LLM Quick Example")
    print("="*60 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Create and save sample data
        raw_df = create_sample_data(n_users=50, events_per_user=30)
        raw_path = tmpdir / 'raw_events.parquet'
        raw_df.to_parquet(raw_path)

        # Step 2: Build training dataset
        print("\n⚙️  Building training dataset...")
        output_path = tmpdir / 'features.parquet'
        train_df, test_df = build_training_dataset(
            raw_events_path=raw_path,
            output_path=output_path,
            test_split=0.2
        )
        print(f"   ✓ Training samples: {len(train_df)}")
        print(f"   ✓ Test samples: {len(test_df)}")

        # Step 3: Train models
        print("\n🤖 Training models...")

        click_model = BaseClickModel()
        click_metrics = click_model.train(
            train_df,
            target_col='clicked',
            valid_df=test_df,
            verbose=False
        )
        print(f"   ✓ Click model - AUC: {click_metrics.get('valid_auc', 0):.3f}")

        complaint_model = BaseComplaintModel()
        # Skip validation if test set has only one class
        if test_df['unsubscribed'].nunique() < 2:
            complaint_metrics = complaint_model.train(
                train_df,
                target_col='unsubscribed',
                valid_df=None,
                verbose=False
            )
            print(f"   ✓ Complaint model - Train AUC: {complaint_metrics.get('train_auc', 0):.3f} (no validation - single class in test)")
        else:
            complaint_metrics = complaint_model.train(
                train_df,
                target_col='unsubscribed',
                valid_df=test_df,
                verbose=False
            )
            print(f"   ✓ Complaint model - AUC: {complaint_metrics.get('valid_auc', 0):.3f}")

        # Step 4: Make a single decision
        print("\n📝 Making a single decision...")

        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=FatigueModel(),
            llm_client=MockLLMClient()
        )

        # Create a test candidate
        candidate = CandidateNotification(
            user_id=1,
            template_id='promo_weekend',
            channel=Channel.PUSH,
            notification_type=NotificationType.PROMO,
            text='Weekend Special: 20% off!',
            timestamp=datetime.now(),
            hour_of_day=14,
            day_of_week=5,
            user_features=UserFeatures(
                user_id=1,
                timestamp=datetime.now(),
                notif_count_24h=2,
                notif_count_7d=10,
                notif_count_30d=35,
                open_rate_7d=0.15,
                open_rate_30d=0.14
            )
        )

        decision = decision_maker.decide_single(candidate, use_llm=False)

        print(f"   Decision: {'SEND ✓' if decision.should_send else 'REJECT ✗'}")
        print(f"   Final score: {decision.scored.final_score:.3f}")
        print(f"   P(click): {decision.scored.p_click:.3f}")
        print(f"   P(complaint): {decision.scored.p_complaint:.3f}")
        print(f"   Reason: {decision.reason}")

        # Step 5: Run offline simulation
        print("\n🎯 Running offline simulation...")

        simulator = NotificationSimulator(decision_maker)
        result = simulator.simulate(test_df, use_llm=False)

        print(f"   ✓ Processed {result.total_candidates} candidates")
        print(f"   ✓ Sent {result.sent} ({result.sent/result.total_candidates:.1%})")

        # Compute metrics
        metrics = MetricsCalculator.compute_metrics(result)

        print("\n📊 Performance Metrics:")
        print(f"   Precision: {metrics.precision:.2%}")
        print(f"   Recall: {metrics.recall:.2%}")
        print(f"   F1 Score: {metrics.f1_score:.2%}")
        print(f"   Click-through rate: {metrics.click_through_rate:.2%}")
        print(f"   Complaint rate: {metrics.complaint_rate:.2%}")

        print("\n" + "="*60)
        print("✅ Example completed successfully!")
        print("="*60 + "\n")

        print("💡 Next steps:")
        print("   - Run 'python tinynotify.py api' to start the API server")
        print("   - Run 'python tinynotify.py dashboard' to launch the UI")
        print("   - See examples/ folder for API request examples")
        print()


if __name__ == '__main__':
    main()
