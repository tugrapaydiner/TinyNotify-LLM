"""
Base prediction models for click and complaint likelihood.
Uses LightGBM for efficient CPU-only inference.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pickle
import json
from datetime import datetime

from src.config import Config, DEFAULT_CONFIG
from src.features.build_dataset import get_feature_columns


class BaseClickModel:
    """
    Predicts probability that user will click on notification.

    Uses LightGBM binary classifier optimized for CPU inference.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize click prediction model.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: List[str] = get_feature_columns(self.config)
        self.training_metadata: Dict = {}

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str = 'clicked',
        valid_df: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the click prediction model.

        Args:
            train_df: Training data with features and target
            target_col: Name of target column
            valid_df: Optional validation data
            verbose: Whether to print training progress

        Returns:
            Dictionary with training metrics
        """
        # Validate required columns
        assert target_col in train_df.columns, f"Missing target column: {target_col}"
        for col in self.feature_columns:
            assert col in train_df.columns, f"Missing feature column: {col}"

        # Extract features and target
        X_train = train_df[self.feature_columns].values
        y_train = train_df[target_col].astype(int).values

        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_columns
        )

        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']

        if valid_df is not None:
            X_valid = valid_df[self.feature_columns].values
            y_valid = valid_df[target_col].astype(int).values
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                feature_name=self.feature_columns,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # Get model parameters from config
        params = self.config.model.click_model_params.copy()

        # Extract n_estimators (it's in params dict)
        n_estimators = params.pop('n_estimators', 100)

        # Train model
        if verbose:
            print(f"Training click model with {len(X_train)} examples...")
            print(f"Positive rate: {y_train.mean():.4f}")

        # Early stopping callback if validation set provided
        callbacks = None
        if valid_df is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # Store training metadata
        self.training_metadata = {
            'train_timestamp': datetime.now().isoformat(),
            'n_train_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'target_positive_rate': float(y_train.mean()),
            'best_iteration': self.model.best_iteration if valid_df is not None else self.model.num_trees(),
        }

        # Compute metrics
        metrics = self._compute_metrics(train_df, target_col, 'train')

        if valid_df is not None:
            valid_metrics = self._compute_metrics(valid_df, target_col, 'valid')
            metrics.update(valid_metrics)

        self.training_metadata['metrics'] = metrics

        if verbose:
            print(f"Training complete. Best iteration: {self.training_metadata['best_iteration']}")
            print(f"Train AUC: {metrics.get('train_auc', 0):.4f}")
            if 'valid_auc' in metrics:
                print(f"Valid AUC: {metrics['valid_auc']:.4f}")

        return metrics

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        target_col: str,
        prefix: str
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc

        X = df[self.feature_columns].values
        y_true = df[target_col].astype(int).values
        y_pred_proba = self.model.predict(X)

        # Compute metrics
        metrics = {}

        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            metrics[f'{prefix}_auc'] = roc_auc_score(y_true, y_pred_proba)

        # Log loss
        metrics[f'{prefix}_logloss'] = log_loss(y_true, y_pred_proba)

        # PR-AUC
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics[f'{prefix}_pr_auc'] = auc(recall, precision)

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict click probabilities.

        Args:
            df: DataFrame with features

        Returns:
            Array of click probabilities
        """
        assert self.model is not None, "Model not trained. Call train() first."

        # Validate features
        for col in self.feature_columns:
            assert col in df.columns, f"Missing feature column: {col}"

        X = df[self.feature_columns].values
        predictions = self.model.predict(X)

        return predictions

    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Predict click probability for single instance.

        Args:
            features: Dictionary mapping feature names to values

        Returns:
            Click probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Fill missing features with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return float(self.predict(df)[0])

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model (will create .txt for model and .json for metadata)
        """
        assert self.model is not None, "No model to save. Train first."

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_path = path.with_suffix('.txt')
        self.model.save_model(str(model_path))

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'training_metadata': self.training_metadata,
            'model_type': 'click'
        }

        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file (without extension)
        """
        path = Path(path)

        # Load LightGBM model
        model_path = path.with_suffix('.txt')
        assert model_path.exists(), f"Model file not found: {model_path}"
        self.model = lgb.Booster(model_file=str(model_path))

        # Load metadata
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata['feature_columns']
                self.training_metadata = metadata['training_metadata']

        print(f"Model loaded from {model_path}")

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight')

        Returns:
            DataFrame with feature names and importance scores
        """
        assert self.model is not None, "Model not trained."

        importance = self.model.feature_importance(importance_type=importance_type)

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)


class BaseComplaintModel:
    """
    Predicts probability of complaint/unsubscribe.

    Uses LightGBM binary classifier optimized for CPU inference.
    Handles class imbalance (complaints are rare).
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize complaint prediction model.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: List[str] = get_feature_columns(self.config)
        self.training_metadata: Dict = {}

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str = 'unsubscribed',
        valid_df: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the complaint prediction model.

        Args:
            train_df: Training data with features and target
            target_col: Name of target column
            valid_df: Optional validation data
            verbose: Whether to print training progress

        Returns:
            Dictionary with training metrics
        """
        # Validate required columns
        assert target_col in train_df.columns, f"Missing target column: {target_col}"
        for col in self.feature_columns:
            assert col in train_df.columns, f"Missing feature column: {col}"

        # Extract features and target
        X_train = train_df[self.feature_columns].values
        y_train = train_df[target_col].astype(int).values

        # Compute class weight for imbalanced data
        pos_rate = y_train.mean()
        if pos_rate > 0 and pos_rate < 1:
            scale_pos_weight = (1 - pos_rate) / pos_rate
        else:
            scale_pos_weight = 1.0

        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_columns
        )

        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']

        if valid_df is not None:
            X_valid = valid_df[self.feature_columns].values
            y_valid = valid_df[target_col].astype(int).values
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                feature_name=self.feature_columns,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # Get model parameters from config and add scale_pos_weight
        params = self.config.model.complaint_model_params.copy()
        params['scale_pos_weight'] = scale_pos_weight

        # Extract n_estimators (it's in params dict)
        n_estimators = params.pop('n_estimators', 50)

        # Train model
        if verbose:
            print(f"Training complaint model with {len(X_train)} examples...")
            print(f"Positive rate: {y_train.mean():.4f}")
            print(f"Scale pos weight: {scale_pos_weight:.2f}")

        # Early stopping callback if validation set provided
        callbacks = None
        if valid_df is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # Store training metadata
        self.training_metadata = {
            'train_timestamp': datetime.now().isoformat(),
            'n_train_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'target_positive_rate': float(y_train.mean()),
            'scale_pos_weight': float(scale_pos_weight),
            'best_iteration': self.model.best_iteration if valid_df is not None else self.model.num_trees(),
        }

        # Compute metrics
        metrics = self._compute_metrics(train_df, target_col, 'train')

        if valid_df is not None:
            valid_metrics = self._compute_metrics(valid_df, target_col, 'valid')
            metrics.update(valid_metrics)

        self.training_metadata['metrics'] = metrics

        if verbose:
            print(f"Training complete. Best iteration: {self.training_metadata['best_iteration']}")
            print(f"Train AUC: {metrics.get('train_auc', 0):.4f}")
            if 'valid_auc' in metrics:
                print(f"Valid AUC: {metrics['valid_auc']:.4f}")

        return metrics

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        target_col: str,
        prefix: str
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc

        X = df[self.feature_columns].values
        y_true = df[target_col].astype(int).values
        y_pred_proba = self.model.predict(X)

        # Compute metrics
        metrics = {}

        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            metrics[f'{prefix}_auc'] = roc_auc_score(y_true, y_pred_proba)

        # Log loss
        metrics[f'{prefix}_logloss'] = log_loss(y_true, y_pred_proba)

        # PR-AUC (important for imbalanced data)
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics[f'{prefix}_pr_auc'] = auc(recall, precision)

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict complaint probabilities.

        Args:
            df: DataFrame with features

        Returns:
            Array of complaint probabilities
        """
        assert self.model is not None, "Model not trained. Call train() first."

        # Validate features
        for col in self.feature_columns:
            assert col in df.columns, f"Missing feature column: {col}"

        X = df[self.feature_columns].values
        predictions = self.model.predict(X)

        return predictions

    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Predict complaint probability for single instance.

        Args:
            features: Dictionary mapping feature names to values

        Returns:
            Complaint probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Fill missing features with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return float(self.predict(df)[0])

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model (will create .txt for model and .json for metadata)
        """
        assert self.model is not None, "No model to save. Train first."

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_path = path.with_suffix('.txt')
        self.model.save_model(str(model_path))

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'training_metadata': self.training_metadata,
            'model_type': 'complaint'
        }

        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file (without extension)
        """
        path = Path(path)

        # Load LightGBM model
        model_path = path.with_suffix('.txt')
        assert model_path.exists(), f"Model file not found: {model_path}"
        self.model = lgb.Booster(model_file=str(model_path))

        # Load metadata
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata['feature_columns']
                self.training_metadata = metadata['training_metadata']

        print(f"Model loaded from {model_path}")

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight')

        Returns:
            DataFrame with feature names and importance scores
        """
        assert self.model is not None, "Model not trained."

        importance = self.model.feature_importance(importance_type=importance_type)

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
