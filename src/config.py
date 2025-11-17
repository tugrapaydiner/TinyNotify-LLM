"""
Global configuration and hyperparameters for TinyNotify-LLM.
All settings are centralized here for easy experimentation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os


@dataclass
class PathConfig:
    """File paths configuration."""
    project_root: Path = Path(__file__).parent.parent
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_raw = self.project_root / "data" / "raw"
        self.data_processed = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"

        # Create directories if they don't exist
        for path in [self.data_raw, self.data_processed,
                     self.models_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Time windows for user statistics (in hours)
    window_24h: int = 24
    window_7d: int = 168  # 7 * 24
    window_30d: int = 720  # 30 * 24

    # Feature dimensions
    template_hash_dim: int = 2048
    max_user_id: int = 1_000_000

    # Time buckets for hour-of-day features
    time_buckets: List[str] = field(default_factory=lambda: [
        'night',      # 0-6
        'morning',    # 6-12
        'afternoon',  # 12-18
        'evening'     # 18-24
    ])

    # Channels
    channels: List[str] = field(default_factory=lambda: [
        'email', 'push', 'sms', 'in_app'
    ])

    # Notification types
    notification_types: List[str] = field(default_factory=lambda: [
        'promo', 'product_tip', 'important', 'system_alert'
    ])


@dataclass
class ModelConfig:
    """Base model training configuration."""
    # Click model
    click_model_type: str = "lightgbm"  # or "logistic"
    click_model_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })

    # Complaint/fatigue model
    complaint_model_type: str = "lightgbm"
    complaint_model_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 50,
        'min_child_samples': 20,
        'class_weight': 'balanced',  # Handle class imbalance
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })

    # Train/validation split
    validation_split: float = 0.2
    random_state: int = 42


@dataclass
class LLMConfig:
    """Tiny LLM configuration."""
    # Model path (user must download and set this)
    model_path: Optional[str] = None

    # Model constraints
    max_context_length: int = 256
    max_tokens_output: int = 8
    temperature: float = 0.1  # Low temperature for deterministic ratings
    top_p: float = 0.9

    # Performance settings
    n_threads: int = 4
    n_batch: int = 1

    # Safety settings
    timeout_seconds: float = 2.0
    max_retries: int = 2
    fallback_rating: int = 3  # Neutral rating if LLM fails

    # Quantization
    use_4bit: bool = True

    def __post_init__(self):
        # Try to get model path from environment
        if self.model_path is None:
            self.model_path = os.getenv('TINYNOTIFY_LLM_MODEL_PATH')


@dataclass
class PolicyConfig:
    """Notification decision policy configuration."""
    # Score combination weights
    alpha: float = 0.7  # Weight for base model score
    beta: float = 0.3   # Weight for LLM rating

    # Complaint penalty in base score
    lambda_comp: float = 5.0

    # Decision threshold
    score_threshold: float = 0.4

    # Top-K candidates to send to LLM
    top_k_for_llm: int = 3

    # Rate limiting
    max_notifications_per_day: int = 5
    max_notifications_per_hour: int = 2

    # Quiet hours (no notifications except critical)
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7     # 7 AM
    allow_critical_in_quiet_hours: bool = True

    # Per-type limits
    max_promo_per_week: int = 3
    max_tips_per_week: int = 7


@dataclass
class SimulationConfig:
    """Simulation and evaluation configuration."""
    # Metrics to compute
    compute_ctr: bool = True
    compute_volume: bool = True
    compute_unsub_rate: bool = True
    compute_fatigue: bool = True

    # Fatigue metric weight
    gamma_fatigue: float = 0.01

    # Replay settings
    batch_size: int = 1000
    enable_progress_bar: bool = True


@dataclass
class ResourceConstraints:
    """Resource constraint verification."""
    max_memory_mb: int = 2048  # 2 GB
    max_latency_ms: float = 500.0  # Per decision
    cpu_only: bool = True

    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        return mem_mb

    def verify_memory_constraint(self) -> bool:
        """Verify we're under memory limit."""
        current = self.check_memory_usage()
        return current <= self.max_memory_mb


@dataclass
class Config:
    """Master configuration object."""
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    resources: ResourceConstraints = field(default_factory=ResourceConstraints)

    def save_to_file(self, filepath: Path):
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict

        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: Path):
        """Load configuration from JSON file."""
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested dataclasses
        # (Simplified - in production, implement proper deserialization)
        return cls()


# Global default configuration
DEFAULT_CONFIG = Config()
