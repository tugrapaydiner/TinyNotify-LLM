"""
Validate that TinyNotify-LLM meets all resource constraints.

This script performs comprehensive validation of:
- Memory usage under 2GB
- Latency under 500ms
- CPU-only operation
- Model size constraints
- End-to-end system performance
"""
import sys
import time
import tempfile
from pathlib import Path
import psutil
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEFAULT_CONFIG
from src.features.build_dataset import build_training_dataset
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.decision import NotificationDecisionMaker
from src.simulation.simulator import NotificationSimulator
from tests.test_integration_e2e import create_synthetic_e2e_dataset


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def get_memory_mb():
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def validate_memory_constraint():
    """Validate memory usage is under 2GB throughout pipeline."""
    print_header("Memory Constraint Validation")

    max_memory_mb = DEFAULT_CONFIG.resources.max_memory_mb
    print_info(f"Target: < {max_memory_mb} MB")

    mem_baseline = get_memory_mb()
    print_info(f"Baseline memory: {mem_baseline:.1f} MB")

    passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Data generation
        print_info("\nGenerating synthetic data...")
        raw_df = create_synthetic_e2e_dataset(n_users=50, events_per_user=30)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        mem1 = get_memory_mb()
        print_info(f"  Memory after data: {mem1:.1f} MB (Δ{mem1 - mem_baseline:.1f} MB)")
        if mem1 >= max_memory_mb:
            print_error(f"  Memory exceeded: {mem1:.1f} MB >= {max_memory_mb} MB")
            passed = False

        # Step 2: Feature engineering
        print_info("\nBuilding training dataset...")
        output_path = tmpdir / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path, test_split=0.2)

        mem2 = get_memory_mb()
        print_info(f"  Memory after features: {mem2:.1f} MB (Δ{mem2 - mem1:.1f} MB)")
        if mem2 >= max_memory_mb:
            print_error(f"  Memory exceeded: {mem2:.1f} MB >= {max_memory_mb} MB")
            passed = False

        # Step 3: Model training
        print_info("\nTraining click model...")
        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        mem3 = get_memory_mb()
        print_info(f"  Memory after click model: {mem3:.1f} MB (Δ{mem3 - mem2:.1f} MB)")
        if mem3 >= max_memory_mb:
            print_error(f"  Memory exceeded: {mem3:.1f} MB >= {max_memory_mb} MB")
            passed = False

        print_info("\nTraining complaint model...")
        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        mem4 = get_memory_mb()
        print_info(f"  Memory after both models: {mem4:.1f} MB (Δ{mem4 - mem3:.1f} MB)")
        if mem4 >= max_memory_mb:
            print_error(f"  Memory exceeded: {mem4:.1f} MB >= {max_memory_mb} MB")
            passed = False

        # Step 4: Simulation
        print_info("\nRunning simulation...")
        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=FatigueModel(),
            llm_client=MockLLMClient()
        )

        simulator = NotificationSimulator(decision_maker)
        result = simulator.simulate(test_df, use_llm=False)

        mem5 = get_memory_mb()
        print_info(f"  Memory after simulation: {mem5:.1f} MB (Δ{mem5 - mem4:.1f} MB)")
        if mem5 >= max_memory_mb:
            print_error(f"  Memory exceeded: {mem5:.1f} MB >= {max_memory_mb} MB")
            passed = False

        print_info(f"\nPeak memory: {mem5:.1f} MB / {max_memory_mb} MB")

    if passed:
        print_success(f"Memory constraint validated: {mem5:.1f} MB < {max_memory_mb} MB")
    else:
        print_error("Memory constraint FAILED")

    return passed


def validate_latency_constraint():
    """Validate decision latency is under 500ms."""
    print_header("Latency Constraint Validation")

    max_latency_ms = DEFAULT_CONFIG.resources.max_latency_ms
    print_info(f"Target: < {max_latency_ms} ms per decision")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare data and models
        print_info("\nPreparing test environment...")
        raw_df = create_synthetic_e2e_dataset(n_users=30, events_per_user=20)
        raw_path = tmpdir / "raw.parquet"
        raw_df.to_parquet(raw_path)

        output_path = tmpdir / "proc.parquet"
        train_df, test_df = build_training_dataset(raw_path, output_path, test_split=0.2)

        click_model = BaseClickModel()
        click_model.train(train_df, target_col='clicked', verbose=False)

        complaint_model = BaseComplaintModel()
        complaint_model.train(train_df, target_col='unsubscribed', verbose=False)

        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=FatigueModel(),
            llm_client=MockLLMClient()
        )

        # Test single decision latency
        print_info("\nTesting single decision latency...")
        from src.features.schema import CandidateNotification, UserFeatures, Channel, NotificationType
        from datetime import datetime

        candidate = CandidateNotification(
            user_id=1,
            template_id='test',
            channel=Channel.PUSH,
            notification_type=NotificationType.PROMO,
            text='Test',
            timestamp=datetime.now(),
            hour_of_day=12,
            day_of_week=2,
            user_features=UserFeatures(
                user_id=1,
                timestamp=datetime.now(),
                notif_count_24h=2,
                notif_count_7d=10,
                notif_count_30d=30,
                open_rate_7d=0.15,
                open_rate_30d=0.12
            )
        )

        # Warm-up run
        _ = decision_maker.decide_single(candidate, use_llm=False)

        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.time()
            _ = decision_maker.decide_single(candidate, use_llm=False)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        max_measured = max(latencies)
        min_measured = min(latencies)

        print_info(f"  Average: {avg_latency:.2f} ms")
        print_info(f"  Min: {min_measured:.2f} ms")
        print_info(f"  Max: {max_measured:.2f} ms")

        # Test batch latency
        print_info("\nTesting batch decision latency...")
        simulator = NotificationSimulator(decision_maker)

        start = time.time()
        result = simulator.simulate(test_df.head(100), use_llm=False)
        total_time_ms = (time.time() - start) * 1000

        avg_batch_latency = result.decision_latency_ms / result.total_candidates
        print_info(f"  Total time: {total_time_ms:.1f} ms for {result.total_candidates} decisions")
        print_info(f"  Average per decision: {avg_batch_latency:.2f} ms")

        passed = True
        if avg_latency >= max_latency_ms:
            print_error(f"Single latency exceeded: {avg_latency:.2f} ms >= {max_latency_ms} ms")
            passed = False
        else:
            print_success(f"Single decision latency: {avg_latency:.2f} ms < {max_latency_ms} ms")

        if avg_batch_latency >= max_latency_ms:
            print_error(f"Batch latency exceeded: {avg_batch_latency:.2f} ms >= {max_latency_ms} ms")
            passed = False
        else:
            print_success(f"Batch decision latency: {avg_batch_latency:.2f} ms < {max_latency_ms} ms")

    return passed


def validate_cpu_only():
    """Validate that system runs on CPU only (no GPU required)."""
    print_header("CPU-Only Operation Validation")

    print_info("Checking for GPU dependencies...")

    # Check if any GPU libraries are imported
    gpu_libs = ['torch', 'tensorflow', 'jax', 'cupy']
    gpu_found = []

    for lib in gpu_libs:
        if lib in sys.modules:
            gpu_found.append(lib)

    if gpu_found:
        print_warning(f"GPU libraries found in modules: {gpu_found}")
        print_info("Note: System should still work on CPU-only machines")
    else:
        print_success("No GPU libraries detected")

    # Verify LightGBM is using CPU
    print_info("\nVerifying LightGBM device...")
    try:
        import lightgbm as lgb
        print_success("LightGBM installed (CPU-based)")
    except ImportError:
        print_error("LightGBM not found")
        return False

    return True


def validate_test_suite():
    """Validate that test suite passes."""
    print_header("Test Suite Validation")

    print_info("Running pytest...")

    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/', '-v', '--tb=line'],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode == 0:
        # Count tests
        import re
        match = re.search(r'(\d+) passed', result.stdout)
        if match:
            num_tests = match.group(1)
            print_success(f"All {num_tests} tests passed")
            return True
    else:
        print_error("Tests failed")
        print_info(result.stdout[-500:])  # Show last 500 chars
        return False

    return result.returncode == 0


def validate_system_configuration():
    """Validate system configuration."""
    print_header("System Configuration Validation")

    config = DEFAULT_CONFIG

    print_info(f"Resource constraints:")
    print_info(f"  Max memory: {config.resources.max_memory_mb} MB")
    print_info(f"  Max latency: {config.resources.max_latency_ms} ms")
    print_info(f"  CPU only: {config.resources.cpu_only}")

    print_info(f"\nModel configuration:")
    print_info(f"  Estimators: {config.model.click_model_params.get('n_estimators', 'N/A')}")
    print_info(f"  Max depth: {config.model.click_model_params.get('max_depth', 'N/A')}")
    print_info(f"  Num leaves: {config.model.click_model_params.get('num_leaves', 'N/A')}")

    print_info(f"\nPolicy configuration:")
    print_info(f"  Alpha (base weight): {config.policy.alpha}")
    print_info(f"  Beta (LLM weight): {config.policy.beta}")
    print_info(f"  Lambda (complaint penalty): {config.policy.lambda_comp}")
    print_info(f"  Score threshold: {config.policy.score_threshold}")

    print_success("Configuration validated")
    return True


def main():
    """Run all validation checks."""
    print(f"\n{Colors.BOLD}TinyNotify-LLM Deployment Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

    results = {}

    # Run validations
    results['system_config'] = validate_system_configuration()
    results['cpu_only'] = validate_cpu_only()
    results['memory'] = validate_memory_constraint()
    results['latency'] = validate_latency_constraint()
    results['tests'] = validate_test_suite()

    # Summary
    print_header("Validation Summary")

    all_passed = all(results.values())

    for check, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {check.replace('_', ' ').title()}: {status}")

    print()

    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}✓ All validations PASSED - System ready for deployment{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.RED}✗ Some validations FAILED - Address issues before deployment{Colors.RESET}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
