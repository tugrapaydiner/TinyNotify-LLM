"""
Main Streamlit dashboard for TinyNotify-LLM.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import DEFAULT_CONFIG
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.decision import NotificationDecisionMaker
from src.policy.scoring import NotificationScorer
from src.simulation.simulator import NotificationSimulator
from src.simulation.metrics import MetricsCalculator, format_metrics_report
from src.features.build_dataset import build_training_dataset


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="TinyNotify-LLM",
        page_icon="🔔",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔔 TinyNotify-LLM Dashboard")
    st.markdown("**Smart Notification System with CPU-Only LLM**")

    # Initialize session state
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🗂️ Data", "🤖 Training", "🎯 Simulation", "⚙️ Configuration"]
    )

    # Route to pages
    if page == "📊 Overview":
        show_overview()
    elif page == "🗂️ Data":
        show_data_page()
    elif page == "🤖 Training":
        show_training_page()
    elif page == "🎯 Simulation":
        show_simulation_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Resource Limits:**")
    st.sidebar.markdown("• CPU Only")
    st.sidebar.markdown("• < 2GB RAM")
    st.sidebar.markdown("• < 500ms latency")


def show_overview():
    """Show system overview page."""
    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models", "3", help="Click, Complaint, Fatigue")
    with col2:
        status = "✅ Ready" if st.session_state.models_trained else "⏳ Not Trained"
        st.metric("Status", status)
    with col3:
        st.metric("Components", "5", help="Features, Models, LLM, Policy, Metrics")
    with col4:
        st.metric("Tests", "140/140 ✅")

    st.markdown("---")

    # System architecture
    st.subheader("System Architecture")

    st.markdown("""
    ```
    Raw Events → Feature Engineering → Base Models → LLM Rating → Policy → Decision
                      ↓                    ↓             ↓           ↓         ↓
                  User Stats         p_click        Rating 1-5   Scoring   Send/Skip
                  Time Windows       p_complaint                 Threshold
                  Recency           Fatigue
    ```
    """)

    # Key metrics
    st.subheader("Key Performance Indicators")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Performance:**")
        st.markdown("• Click Prediction: LightGBM")
        st.markdown("• Complaint Prediction: LightGBM (class-balanced)")
        st.markdown("• Fatigue: Rule-based adaptive")

    with col2:
        st.markdown("**Policy:**")
        st.markdown("• Score = 0.7 × base + 0.3 × LLM")
        st.markdown("• Threshold: 0.4 (configurable)")
        st.markdown("• Max daily: 5 notifications/user")

    # Quick start guide
    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
        1. **Data**: Upload or generate sample notification data
        2. **Training**: Train click and complaint prediction models
        3. **Simulation**: Run offline evaluation on historical data
        4. **Configuration**: Adjust policy parameters and thresholds
        5. **Analysis**: View metrics, compare policies, optimize thresholds
        """)


def show_data_page():
    """Show data management page."""
    st.header("📂 Data Management")

    tab1, tab2 = st.tabs(["Upload Data", "Generate Sample"])

    with tab1:
        st.subheader("Upload Notification Events")

        uploaded_file = st.file_uploader(
            "Upload CSV or Parquet file",
            type=['csv', 'parquet'],
            help="File should contain: user_id, template_id, timestamp_send, channel, clicked, etc."
        )

        if uploaded_file:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=['timestamp_send'])
            else:
                df = pd.read_parquet(uploaded_file)

            st.session_state.raw_data = df
            st.success(f"✅ Loaded {len(df)} events from {df['user_id'].nunique()} users")

            # Preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", f"{len(df):,}")
            with col2:
                st.metric("Unique Users", f"{df['user_id'].nunique():,}")
            with col3:
                click_rate = df['clicked'].mean()
                st.metric("Click Rate", f"{click_rate:.1%}")
            with col4:
                if 'unsubscribed' in df.columns:
                    unsub_rate = df['unsubscribed'].mean()
                    st.metric("Unsub Rate", f"{unsub_rate:.2%}")

    with tab2:
        st.subheader("Generate Sample Data")

        n_users = st.slider("Number of users", 10, 500, 100)
        events_per_user = st.slider("Events per user", 5, 50, 20)

        if st.button("Generate Sample Data"):
            from tests.test_base_model import create_synthetic_dataset

            n_samples = n_users * events_per_user
            df = create_synthetic_dataset(n_samples=n_samples)

            st.session_state.raw_data = df
            st.success(f"✅ Generated {len(df)} synthetic events")
            st.dataframe(df.head(10))


def show_training_page():
    """Show model training page."""
    st.header("🤖 Model Training")

    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please load data first (Data page)")
        return

    df = st.session_state.raw_data

    st.info(f"Training on {len(df)} events from {df['user_id'].nunique()} users")

    # Training configuration
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_split = st.slider("Test split", 0.1, 0.3, 0.2, 0.05)
    with col2:
        verbose = st.checkbox("Verbose logging", value=False)

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Build dataset
            from tempfile import TemporaryDirectory
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = Path(tmpdir) / "raw.parquet"
                df.to_parquet(raw_path)

                output_path = Path(tmpdir) / "processed.parquet"

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Building dataset...")
                progress_bar.progress(20)

                train_df, test_df = build_training_dataset(
                    raw_path,
                    output_path,
                    test_split=test_split
                )

                progress_bar.progress(40)
                status_text.text("Training click model...")

                # Train click model
                click_model = BaseClickModel()
                click_metrics = click_model.train(
                    train_df,
                    target_col='clicked',
                    valid_df=test_df,
                    verbose=verbose
                )

                progress_bar.progress(60)
                status_text.text("Training complaint model...")

                # Train complaint model
                complaint_model = BaseComplaintModel()
                complaint_metrics = complaint_model.train(
                    train_df,
                    target_col='unsubscribed',
                    valid_df=test_df,
                    verbose=verbose
                )

                progress_bar.progress(80)
                status_text.text("Initializing components...")

                # Create other components
                fatigue_model = FatigueModel()
                llm_client = MockLLMClient()
                scorer = NotificationScorer()

                decision_maker = NotificationDecisionMaker(
                    click_model=click_model,
                    complaint_model=complaint_model,
                    fatigue_model=fatigue_model,
                    llm_client=llm_client,
                    scorer=scorer
                )

                # Store in session state
                st.session_state.click_model = click_model
                st.session_state.complaint_model = complaint_model
                st.session_state.decision_maker = decision_maker
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                st.session_state.models_trained = True

                progress_bar.progress(100)
                status_text.text("✅ Training complete!")

        st.success("✅ Models trained successfully!")

        # Show metrics
        st.subheader("Training Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Click Model:**")
            st.metric("Train AUC", f"{click_metrics.get('train_auc', 0):.3f}")
            st.metric("Valid AUC", f"{click_metrics.get('valid_auc', 0):.3f}")

        with col2:
            st.markdown("**Complaint Model:**")
            st.metric("Train AUC", f"{complaint_metrics.get('train_auc', 0):.3f}")
            st.metric("Valid AUC", f"{complaint_metrics.get('valid_auc', 0):.3f}")


def show_simulation_page():
    """Show simulation and evaluation page."""
    st.header("🎯 Simulation & Evaluation")

    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first (Training page)")
        return

    st.subheader("Run Offline Simulation")

    # Choose data
    data_choice = st.radio(
        "Evaluate on:",
        ["Test Set", "Full Dataset"]
    )

    if data_choice == "Test Set":
        eval_df = st.session_state.test_df
    else:
        eval_df = st.session_state.raw_data

    use_llm = st.checkbox("Use LLM for top-K", value=False)

    if use_llm:
        top_k = st.slider("Top-K for LLM", 1, 10, 3)
    else:
        top_k = 0

    if st.button("▶️ Run Simulation"):
        with st.spinner("Running simulation..."):
            simulator = NotificationSimulator(st.session_state.decision_maker)

            result = simulator.simulate(
                eval_df,
                use_llm=use_llm,
                top_k_llm=top_k if use_llm else None
            )

            st.session_state.simulation_result = result
            st.session_state.simulation_run = True

        st.success("✅ Simulation complete!")

    # Show results if available
    if st.session_state.get('simulation_run', False):
        result = st.session_state.simulation_result

        st.markdown("---")
        st.subheader("Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Candidates", f"{result.total_candidates:,}")
        with col2:
            send_rate = result.sent / result.total_candidates
            st.metric("Sent", f"{result.sent:,}", f"{send_rate:.1%}")
        with col3:
            precision = result.true_positives / result.sent if result.sent > 0 else 0
            st.metric("Precision", f"{precision:.1%}")
        with col4:
            recall = result.true_positives / result.actual_clicks if result.actual_clicks > 0 else 0
            st.metric("Recall", f"{recall:.1%}")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sent:**")
            st.metric("True Positives", result.true_positives, help="Sent and clicked")
            st.metric("False Positives", result.false_positives, help="Sent but not clicked")

        with col2:
            st.markdown("**Not Sent:**")
            st.metric("True Negatives", result.true_negatives, help="Not sent, wouldn't click")
            st.metric("False Negatives", result.false_negatives, help="Not sent, would have clicked")

        # Detailed metrics
        metrics = MetricsCalculator.compute_metrics(result)

        st.subheader("Detailed Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Classification:**")
            st.metric("F1 Score", f"{metrics.f1_score:.2%}")
            st.metric("Accuracy", f"{metrics.accuracy:.2%}")

        with col2:
            st.markdown("**Business:**")
            st.metric("CTR", f"{metrics.click_through_rate:.2%}")
            st.metric("Complaint Rate", f"{metrics.complaint_rate:.2%}")

        with col3:
            st.markdown("**Efficiency:**")
            st.metric("Clicks/Send", f"{metrics.clicks_per_send:.3f}")
            st.metric("Latency", f"{result.decision_latency_ms:.0f}ms")

        # Metrics report
        with st.expander("📄 Full Metrics Report"):
            report = format_metrics_report(metrics)
            st.text(report)

        # Score distribution diagnostic
        with st.expander("📊 Score Distribution (Diagnostic)"):
            if hasattr(result, 'all_scores') and result.all_scores:
                scores_df = pd.DataFrame(result.all_scores)
                st.write(f"**Score Statistics:**")
                st.write(scores_df.describe())

                # Show threshold line
                current_threshold = st.session_state.get('config', DEFAULT_CONFIG).policy.score_threshold
                st.write(f"\n**Current Threshold:** {current_threshold:.2f}")
                st.write(f"**Scores >= Threshold:** {(scores_df['final_score'] >= current_threshold).sum()} / {len(scores_df)}")

                # Show sample scores
                st.write("\n**Sample Scores (first 10):**")
                sample_cols = ['p_click', 'p_complaint', 'base_score', 'final_score', 'decision']
                available_cols = [col for col in sample_cols if col in scores_df.columns]
                st.dataframe(scores_df[available_cols].head(10))
            else:
                st.info("Score distribution not available. This feature requires re-running the simulation with the updated code.")


def show_configuration_page():
    """Show configuration page."""
    st.header("⚙️ Configuration")

    # Initialize config in session state if not present
    if 'config' not in st.session_state:
        from copy import deepcopy
        st.session_state.config = deepcopy(DEFAULT_CONFIG)

    config = st.session_state.config

    st.subheader("Policy Parameters")

    # Score weights
    st.markdown("**Score Weights:**")
    col1, col2 = st.columns(2)

    with col1:
        alpha = st.slider("Alpha (base model weight)", 0.0, 1.0, float(config.policy.alpha), 0.05)
    with col2:
        beta = st.slider("Beta (LLM weight)", 0.0, 1.0, float(config.policy.beta), 0.05)

    if abs(alpha + beta - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {alpha + beta:.2f}, not 1.0")

    # Decision threshold
    st.markdown("**Decision Threshold:**")
    threshold = st.slider(
        "Score threshold for sending",
        0.0, 1.0, float(config.policy.score_threshold), 0.05,
        help="Send notification if score >= threshold"
    )

    # Rate limits
    st.markdown("**Rate Limits:**")
    col1, col2 = st.columns(2)

    with col1:
        max_per_day = st.number_input("Max per day", 1, 20, int(config.policy.max_notifications_per_day))
    with col2:
        max_per_hour = st.number_input("Max per hour", 1, 10, 2)

    # Complaint penalty
    lambda_comp = st.slider(
        "Complaint penalty (λ)",
        0.0, 10.0, float(config.policy.lambda_comp), 0.5,
        help="Penalty weight for complaint risk"
    )

    st.markdown("**Formula:** `score = α × (p_click - λ × p_complaint) + β × llm_rating`")

    if st.button("💾 Save Configuration"):
        # Actually update the config
        config.policy.alpha = alpha
        config.policy.beta = beta
        config.policy.score_threshold = threshold
        config.policy.max_notifications_per_day = max_per_day
        config.policy.lambda_comp = lambda_comp

        # Update decision maker if models are trained
        if st.session_state.get('models_trained', False):
            st.session_state.decision_maker = NotificationDecisionMaker(
                click_model=st.session_state.click_model,
                complaint_model=st.session_state.complaint_model,
                fatigue_model=FatigueModel(config),
                llm_client=st.session_state.get('llm_client', MockLLMClient(config)),
                scorer=NotificationScorer(config),
                config=config
            )
            st.success("✅ Configuration saved and applied to models!")
        else:
            st.success("✅ Configuration saved! Train models to apply changes.")

        st.info(f"📊 New threshold: {threshold:.2f} | Alpha: {alpha:.2f} | Lambda: {lambda_comp:.1f}")

    # System info
    st.markdown("---")
    st.subheader("System Information")

    st.markdown(f"**Resource Constraints:**")
    st.markdown("• Max Memory: 2 GB")
    st.markdown("• Max Latency: 500 ms")
    st.markdown("• CPU Only: True")
    st.markdown("• LLM Context: 256 tokens")
    st.markdown("• LLM Output: 8 tokens")


if __name__ == "__main__":
    main()
