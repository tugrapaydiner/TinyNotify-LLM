"""Tests for configuration system."""
import pytest
from pathlib import Path
import tempfile
import json

from src.config import (
    Config, PathConfig, ModelConfig, LLMConfig, PolicyConfig,
    FeatureConfig, SimulationConfig, ResourceConstraints
)


def test_config_initialization():
    """Test that default config initializes without errors."""
    config = Config()
    assert config is not None
    assert config.paths is not None
    assert config.features is not None
    assert config.model is not None
    assert config.llm is not None
    assert config.policy is not None


def test_path_config_creates_directories():
    """Test that PathConfig creates necessary directories."""
    config = PathConfig()
    # Verify paths exist
    assert config.data_raw.exists()
    assert config.data_processed.exists()
    assert config.models_dir.exists()
    assert config.logs_dir.exists()


def test_model_config_has_valid_params():
    """Test that model configs have required parameters."""
    config = ModelConfig()

    # Click model checks
    assert 'objective' in config.click_model_params
    assert 'metric' in config.click_model_params
    assert config.click_model_params['objective'] == 'binary'

    # Complaint model checks
    assert 'objective' in config.complaint_model_params
    assert config.complaint_model_params['objective'] == 'binary'


def test_llm_config_constraints():
    """Test LLM configuration has proper constraints."""
    config = LLMConfig()

    assert config.max_context_length <= 256
    assert config.max_tokens_output <= 10
    assert 0 <= config.temperature <= 1
    assert config.timeout_seconds > 0
    assert 1 <= config.fallback_rating <= 5


def test_policy_config_valid_ranges():
    """Test policy config parameters are in valid ranges."""
    config = PolicyConfig()

    assert 0 <= config.alpha <= 1
    assert 0 <= config.beta <= 1
    assert config.lambda_comp > 0
    assert 0 <= config.score_threshold <= 1
    assert config.top_k_for_llm >= 1
    assert config.max_notifications_per_day >= 1
    assert 0 <= config.quiet_hours_start <= 23
    assert 0 <= config.quiet_hours_end <= 23


def test_resource_constraints_check():
    """Test resource constraint verification."""
    config = Config()

    # Should be able to check memory
    memory_mb = config.resources.check_memory_usage()
    assert memory_mb > 0
    assert memory_mb < 10000  # Sanity check

    # Verify method should work
    result = config.resources.verify_memory_constraint()
    assert isinstance(result, bool)


def test_config_serialization():
    """Test config can be saved and loaded."""
    config1 = Config()
    config1.policy.alpha = 0.8
    config1.policy.beta = 0.2

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config1.save_to_file(Path(f.name))

        # Verify file exists and is valid JSON
        with open(f.name, 'r') as rf:
            data = json.load(rf)
            assert 'policy' in data


def test_config_invariants():
    """Test that config maintains mathematical invariants."""
    config = Config()

    # Alpha + beta should be reasonable (not necessarily = 1, but close)
    alpha_beta_sum = config.policy.alpha + config.policy.beta
    assert 0.5 <= alpha_beta_sum <= 2.0, "Alpha + beta should be reasonable"

    # Time windows should be monotonic
    assert config.features.window_24h < config.features.window_7d
    assert config.features.window_7d < config.features.window_30d

    # Hourly limit should be less than daily limit
    assert config.policy.max_notifications_per_hour <= config.policy.max_notifications_per_day
