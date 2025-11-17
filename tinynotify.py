#!/usr/bin/env python
"""
TinyNotify-LLM Command Line Interface

Single command launcher for all system operations.
"""
import sys
import os
import click
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    TinyNotify-LLM - Smart Notification Decision System

    A production-ready notification system with CPU-only LLM integration.
    """
    pass


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--coverage', '-c', is_flag=True, help='Run with coverage report')
@click.option('--slow', is_flag=True, help='Include slow tests')
def test(verbose, coverage, slow):
    """Run the test suite."""
    import subprocess

    click.echo("🧪 Running test suite...")

    cmd = ['python', '-m', 'pytest', 'tests/']

    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')

    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])

    if slow:
        cmd.append('-m')
        cmd.append('slow')
    else:
        cmd.append('-m')
        cmd.append('not slow')

    result = subprocess.run(cmd)

    if result.returncode == 0:
        click.echo(click.style("✅ All tests passed!", fg='green', bold=True))
        if coverage:
            click.echo(f"📊 Coverage report: {Path('htmlcov/index.html').absolute()}")
    else:
        click.echo(click.style("❌ Tests failed!", fg='red', bold=True))
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, type=int, help='Number of workers')
def api(host, port, reload, workers):
    """Start the FastAPI server."""
    import uvicorn

    click.echo(f"🚀 Starting API server on {host}:{port}")
    click.echo(f"📖 API documentation: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info"
    )


@cli.command()
@click.option('--port', default=8501, type=int, help='Port to bind to')
def dashboard(port):
    """Start the Streamlit dashboard."""
    import subprocess

    click.echo(f"📊 Starting Streamlit dashboard on port {port}")
    click.echo(f"🌐 Dashboard URL: http://localhost:{port}")

    subprocess.run([
        'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0'
    ])


@cli.command()
def validate():
    """Run deployment validation checks."""
    import subprocess

    click.echo("🔍 Running deployment validation...")

    result = subprocess.run(['python', 'scripts/validate_constraints.py'])

    if result.returncode == 0:
        click.echo(click.style("\n✅ Validation passed - System ready for deployment!", fg='green', bold=True))
    else:
        click.echo(click.style("\n❌ Validation failed - Address issues before deployment", fg='red', bold=True))
        sys.exit(1)


@cli.command()
@click.argument('raw_data_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/processed/features.parquet', help='Output path for processed features')
@click.option('--test-split', default=0.2, type=float, help='Test split ratio')
def preprocess(raw_data_path, output, test_split):
    """Preprocess raw data into training features."""
    from src.features.build_dataset import build_training_dataset
    from pathlib import Path

    click.echo(f"⚙️  Processing data from {raw_data_path}...")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df = build_training_dataset(
        raw_events_path=Path(raw_data_path),
        output_path=output_path,
        test_split=test_split
    )

    click.echo(f"✅ Data processed successfully!")
    click.echo(f"   Training samples: {len(train_df)}")
    click.echo(f"   Test samples: {len(test_df)}")
    click.echo(f"   Output: {output_path.absolute()}")


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='models/', help='Directory to save trained models')
@click.option('--verbose', '-v', is_flag=True, help='Verbose training output')
def train(data_path, output_dir, verbose):
    """Train click and complaint prediction models."""
    import pandas as pd
    import pickle
    from pathlib import Path
    from src.models.base_model import BaseClickModel, BaseComplaintModel
    from src.config import DEFAULT_CONFIG

    click.echo(f"🤖 Training models on {data_path}...")

    # Load data
    df = pd.read_parquet(data_path)
    click.echo(f"   Loaded {len(df)} samples")

    # Split into train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    click.echo(f"   Training on {len(train_df)} samples")
    click.echo(f"   Validating on {len(test_df)} samples")

    # Train click model
    click.echo("\n📈 Training click prediction model...")
    click_model = BaseClickModel(DEFAULT_CONFIG)
    click_metrics = click_model.train(
        train_df,
        target_col='clicked',
        valid_df=test_df,
        verbose=verbose
    )

    click.echo(f"   ✓ Click model - Train AUC: {click_metrics.get('train_auc', 0):.3f}, Valid AUC: {click_metrics.get('valid_auc', 0):.3f}")

    # Train complaint model
    click.echo("\n📉 Training complaint prediction model...")
    complaint_model = BaseComplaintModel(DEFAULT_CONFIG)
    complaint_metrics = complaint_model.train(
        train_df,
        target_col='unsubscribed',
        valid_df=test_df,
        verbose=verbose
    )

    click.echo(f"   ✓ Complaint model - Train AUC: {complaint_metrics.get('train_auc', 0):.3f}, Valid AUC: {complaint_metrics.get('valid_auc', 0):.3f}")

    # Save models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click_model_path = output_path / 'click_model.pkl'
    complaint_model_path = output_path / 'complaint_model.pkl'

    with open(click_model_path, 'wb') as f:
        pickle.dump(click_model, f)

    with open(complaint_model_path, 'wb') as f:
        pickle.dump(complaint_model, f)

    click.echo(f"\n💾 Models saved to {output_path.absolute()}/")
    click.echo(f"   - click_model.pkl")
    click.echo(f"   - complaint_model.pkl")

    click.echo(click.style("\n✅ Training complete!", fg='green', bold=True))


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--models-dir', default='models/', help='Directory with trained models')
@click.option('--use-llm', is_flag=True, help='Use LLM for scoring')
@click.option('--top-k', default=5, type=int, help='Number of top candidates for LLM')
def simulate(data_path, models_dir, use_llm, top_k):
    """Run offline simulation on test data."""
    import pandas as pd
    import pickle
    from pathlib import Path
    from src.models.base_model import BaseClickModel, BaseComplaintModel
    from src.models.fatigue_model import FatigueModel
    from src.llm.client import MockLLMClient
    from src.policy.decision import NotificationDecisionMaker
    from src.simulation.simulator import NotificationSimulator
    from src.simulation.metrics import MetricsCalculator, format_metrics_report

    click.echo(f"🎯 Running simulation on {data_path}...")

    # Load models
    models_path = Path(models_dir)
    click_model_path = models_path / 'click_model.pkl'
    complaint_model_path = models_path / 'complaint_model.pkl'

    if not click_model_path.exists() or not complaint_model_path.exists():
        click.echo(click.style("❌ Models not found. Run 'tinynotify train' first.", fg='red'))
        sys.exit(1)

    click.echo("   Loading models...")
    with open(click_model_path, 'rb') as f:
        click_model = pickle.load(f)

    with open(complaint_model_path, 'rb') as f:
        complaint_model = pickle.load(f)

    # Initialize decision maker
    decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=FatigueModel(),
        llm_client=MockLLMClient()
    )

    # Load test data
    test_df = pd.read_parquet(data_path)
    click.echo(f"   Loaded {len(test_df)} test samples")

    # Run simulation
    simulator = NotificationSimulator(decision_maker)

    click.echo(f"\n🔄 Running simulation (LLM: {'enabled' if use_llm else 'disabled'})...")
    result = simulator.simulate(test_df, use_llm=use_llm, top_k_llm=top_k if use_llm else None)

    # Compute metrics
    metrics = MetricsCalculator.compute_metrics(result)

    # Display results
    click.echo("\n📊 Simulation Results:")
    click.echo(f"   Total candidates: {result.total_candidates}")
    click.echo(f"   Sent: {result.sent} ({result.sent/result.total_candidates:.1%})")
    click.echo(f"   Rejected: {result.rejected} ({result.rejected/result.total_candidates:.1%})")
    click.echo(f"\n📈 Performance Metrics:")
    click.echo(f"   Precision: {metrics.precision:.2%}")
    click.echo(f"   Recall: {metrics.recall:.2%}")
    click.echo(f"   F1 Score: {metrics.f1_score:.2%}")
    click.echo(f"   Click-through rate: {metrics.click_through_rate:.2%}")
    click.echo(f"   Complaint rate: {metrics.complaint_rate:.2%}")
    click.echo(f"\n⏱️  Performance:")
    click.echo(f"   Avg latency: {result.decision_latency_ms / result.total_candidates:.2f} ms/decision")

    click.echo(click.style("\n✅ Simulation complete!", fg='green', bold=True))


@cli.command()
def health():
    """Check system health and configuration."""
    from src.config import DEFAULT_CONFIG
    import psutil
    import os

    click.echo("🏥 System Health Check\n")

    # System info
    click.echo("💻 System Information:")
    mem = psutil.virtual_memory()
    click.echo(f"   Total RAM: {mem.total / 1024 / 1024 / 1024:.1f} GB")
    click.echo(f"   Available RAM: {mem.available / 1024 / 1024 / 1024:.1f} GB")
    click.echo(f"   CPU count: {os.cpu_count()}")

    # Configuration
    click.echo("\n⚙️  Configuration:")
    click.echo(f"   Max memory: {DEFAULT_CONFIG.resources.max_memory_mb} MB")
    click.echo(f"   Max latency: {DEFAULT_CONFIG.resources.max_latency_ms} ms")
    click.echo(f"   CPU only: {DEFAULT_CONFIG.resources.cpu_only}")

    # Model config
    click.echo("\n🤖 Model Configuration:")
    click.echo(f"   Model type: {DEFAULT_CONFIG.model.click_model_type}")
    click.echo(f"   Estimators: {DEFAULT_CONFIG.model.click_model_params['n_estimators']}")
    click.echo(f"   Max depth: {DEFAULT_CONFIG.model.click_model_params['max_depth']}")

    # Policy config
    click.echo("\n📋 Policy Configuration:")
    click.echo(f"   Alpha (base weight): {DEFAULT_CONFIG.policy.alpha}")
    click.echo(f"   Beta (LLM weight): {DEFAULT_CONFIG.policy.beta}")
    click.echo(f"   Score threshold: {DEFAULT_CONFIG.policy.score_threshold}")
    click.echo(f"   Max notifications/day: {DEFAULT_CONFIG.policy.max_notifications_per_day}")

    # Check dependencies
    click.echo("\n📦 Dependencies:")
    try:
        import numpy
        click.echo(f"   ✓ numpy {numpy.__version__}")
    except ImportError:
        click.echo("   ✗ numpy not installed")

    try:
        import pandas
        click.echo(f"   ✓ pandas {pandas.__version__}")
    except ImportError:
        click.echo("   ✗ pandas not installed")

    try:
        import lightgbm
        click.echo(f"   ✓ lightgbm {lightgbm.__version__}")
    except ImportError:
        click.echo("   ✗ lightgbm not installed")

    try:
        import fastapi
        click.echo(f"   ✓ fastapi {fastapi.__version__}")
    except ImportError:
        click.echo("   ✗ fastapi not installed")

    try:
        import streamlit
        click.echo(f"   ✓ streamlit {streamlit.__version__}")
    except ImportError:
        click.echo("   ✗ streamlit not installed")

    click.echo(click.style("\n✅ Health check complete!", fg='green', bold=True))


@cli.command()
def quickstart():
    """Quick start guide for new users."""
    guide = """
╔════════════════════════════════════════════════════════════════╗
║              TinyNotify-LLM Quick Start Guide                 ║
╚════════════════════════════════════════════════════════════════╝

📚 STEP 1: Run Tests
   $ tinynotify test

   This will verify everything is working correctly.

⚙️  STEP 2: Prepare Your Data
   Place raw notification events in data/raw/events.parquet

   Then preprocess:
   $ tinynotify preprocess data/raw/events.parquet

🤖 STEP 3: Train Models
   $ tinynotify train data/processed/features.parquet

   This creates click and complaint prediction models.

🎯 STEP 4: Run Simulation (Optional)
   $ tinynotify simulate data/processed/features.parquet

   This evaluates model performance offline.

🚀 STEP 5: Start Services

   Option A - Start Everything (Recommended):
   $ tinynotify start

   This launches both API and Dashboard at once!
   • API Docs: http://localhost:8000/docs
   • Dashboard: http://localhost:8501

   Option B - API Server only:
   $ tinynotify api

   Option C - Dashboard only:
   $ tinynotify dashboard

📖 For more information:
   - README.md: Quick reference
   - DEPLOYMENT.md: Deployment guide
   - Claude.md: Complete implementation details

💡 Useful Commands:
   $ tinynotify health      # Check system status
   $ tinynotify validate    # Run deployment validation
   $ tinynotify test -c     # Run tests with coverage

Need help? Run: tinynotify --help
    """
    click.echo(guide)


@cli.command()
@click.option('--api-port', default=8000, type=int, help='API server port')
@click.option('--dashboard-port', default=8501, type=int, help='Dashboard port')
@click.option('--skip-validation', is_flag=True, help='Skip initial validation')
def start(api_port, dashboard_port, skip_validation):
    """
    Start ALL services at once (API + Dashboard).

    This launches both the API server and Streamlit dashboard
    simultaneously for quick testing and demonstration.
    """
    import subprocess
    import time
    import signal
    import atexit

    click.echo("🚀 Starting TinyNotify-LLM - All Services")
    click.echo("=" * 60)

    # Run validation first unless skipped
    if not skip_validation:
        click.echo("\n🔍 Running quick validation...")
        try:
            from src.config import DEFAULT_CONFIG
            import psutil
            import os

            # Quick memory check
            mem = psutil.virtual_memory()
            click.echo(f"   ✓ Available RAM: {mem.available / 1024 / 1024 / 1024:.1f} GB")

            # Quick dependency check
            import lightgbm
            import fastapi
            import streamlit
            click.echo(f"   ✓ Dependencies installed")

            click.echo("   ✓ System ready")
        except Exception as e:
            click.echo(click.style(f"   ⚠ Validation warning: {e}", fg='yellow'))
            if not click.confirm("Continue anyway?"):
                return

    processes = []

    def cleanup():
        """Kill all spawned processes on exit."""
        click.echo("\n\n🛑 Shutting down services...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        click.echo("✅ All services stopped")

    # Register cleanup
    atexit.register(cleanup)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        click.echo("\n\n⚠️  Received interrupt signal...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start API server
        click.echo(f"\n📡 Starting API Server on port {api_port}...")
        api_process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'src.api.server:app',
             '--host', '0.0.0.0',
             '--port', str(api_port),
             '--log-level', 'warning'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(api_process)
        time.sleep(2)  # Give API time to start

        if api_process.poll() is None:
            click.echo(f"   ✓ API Server running")
            click.echo(f"   📖 API Docs: http://localhost:{api_port}/docs")
        else:
            click.echo(click.style("   ✗ API Server failed to start", fg='red'))
            return

        # Start Streamlit dashboard
        click.echo(f"\n📊 Starting Dashboard on port {dashboard_port}...")
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'app.py',
             '--server.port', str(dashboard_port),
             '--server.address', '0.0.0.0',
             '--server.headless', 'true'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(dashboard_process)
        time.sleep(3)  # Give Streamlit time to start

        if dashboard_process.poll() is None:
            click.echo(f"   ✓ Dashboard running")
            click.echo(f"   🌐 Dashboard: http://localhost:{dashboard_port}")
        else:
            click.echo(click.style("   ✗ Dashboard failed to start", fg='red'))
            click.echo("   💡 Tip: Dashboard may need models trained first")

        # Summary
        click.echo("\n" + "=" * 60)
        click.echo(click.style("✅ ALL SERVICES RUNNING", fg='green', bold=True))
        click.echo("=" * 60)

        click.echo("\n🔗 Quick Links:")
        click.echo(f"   • API Documentation: http://localhost:{api_port}/docs")
        click.echo(f"   • Dashboard:         http://localhost:{dashboard_port}")
        click.echo(f"   • API Health:        http://localhost:{api_port}/health")

        click.echo("\n💡 Test the API:")
        click.echo(f"   curl -X POST http://localhost:{api_port}/decide \\")
        click.echo(f"     -H 'Content-Type: application/json' \\")
        click.echo(f"     -d @examples/example_api_request.json")

        click.echo("\n📝 Logs:")
        click.echo("   • API logs: Check terminal or --log-level info")
        click.echo("   • Dashboard logs: Streamlit output above")

        click.echo("\n⚠️  Press Ctrl+C to stop all services")
        click.echo("=" * 60)

        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if processes are still alive
            if api_process.poll() is not None:
                click.echo(click.style("\n⚠️  API Server stopped unexpectedly", fg='red'))
                break

            if dashboard_process.poll() is not None:
                click.echo(click.style("\n⚠️  Dashboard stopped unexpectedly", fg='yellow'))
                # Don't break - API might still be useful

    except KeyboardInterrupt:
        pass
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg='red'))
    finally:
        cleanup()


if __name__ == '__main__':
    cli()
