"""
ML Model Training Pipeline
XGBoost classifier for price direction prediction.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visual_logger import visual

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Train and manage XGBoost models for trading."""
    
    def __init__(self, 
                 model_dir: str = 'ml/models',
                 random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save models
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: XGBoost parameters (uses defaults if None)
            
        Returns:
            Trained XGBClassifier
        """
        if params is None:
            params = self._default_params()
        
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Training XGBoost model with {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        try:
            # Try new XGBoost API (v2.0+)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        except TypeError:
            # Fallback to old API
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
        
        logger.info(f"Training complete.")
        
        return self.model
    
    def cross_validate(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation...")
        
        visual.print_training_header()
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**self._default_params())
            model.fit(X_train, y_train, verbose=False)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Metrics
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
            
            # Visual progress
            visual.print_training_progress(fold + 1, n_splits, acc)
            logger.info(f"Fold {fold + 1}: Accuracy={accuracies[-1]:.3f}, F1={f1_scores[-1]:.3f}")
        
        results = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'precision_mean': np.mean(precisions),
            'recall_mean': np.mean(recalls),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'fold_accuracies': accuracies
        }
        
        logger.info(f"CV Results - Accuracy: {results['accuracy_mean']:.3f} (+/- {results['accuracy_std']*2:.3f})")
        logger.info(f"CV Results - F1: {results['f1_mean']:.3f} (+/- {results['f1_std']*2:.3f})")
        
        return results
    
    def evaluate(self, 
                 X_test: pd.DataFrame, 
                 y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'prediction_distribution': np.bincount(y_pred),
            'mean_confidence': np.mean(np.abs(y_prob - 0.5)) * 2
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Test F1: {metrics['f1']:.3f}")
        logger.info(f"Mean Confidence: {metrics['mean_confidence']:.3f}")
        
        # Visual completion
        visual.print_training_complete(metrics)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            filename: Custom filename (uses timestamp if None)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"xgboost_{timestamp}.pkl"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Trained at: {model_data.get('trained_at', 'Unknown')}")
    
    def _default_params(self) -> Dict:
        """Default XGBoost parameters."""
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
    
    def optimize_hyperparameters(self, 
                                  X_train: pd.DataFrame, 
                                  y_train: pd.Series,
                                  n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        import optuna
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best F1: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params


class ModelInference:
    """Real-time model inference for trading."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved model
        """
        self.model = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model from disk."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Inference model loaded from {model_path}")
    
    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction on features.
        
        Args:
            features: DataFrame with features (single row or multiple)
            
        Returns:
            (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Load model first")
        
        # Ensure correct feature order
        features = features[self.feature_names]
        
        # Predict
        prediction = self.model.predict(features)[-1]
        probability = self.model.predict_proba(features)[:, 1][-1]
        
        return int(prediction), float(probability)
    
    def predict_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on batch of data.
        
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Load model first")
        
        features = features[self.feature_names]
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        results = pd.DataFrame({
            'prediction': predictions,
            'probability_up': probabilities,
            'confidence': np.abs(probabilities - 0.5) * 2
        }, index=features.index)
        
        return results
    
    def get_confidence_threshold(self, 
                                  features: pd.DataFrame, 
                                  target_accuracy: float = 0.65) -> float:
        """
        Find confidence threshold to achieve target accuracy.
        
        Args:
            features: Feature DataFrame
            target_accuracy: Target accuracy level
            
        Returns:
            Confidence threshold
        """
        results = self.predict_batch(features)
        
        # Sort by confidence
        sorted_conf = results['confidence'].sort_values(ascending=False)
        
        # Find threshold that keeps top X% with target accuracy
        threshold_idx = int(len(sorted_conf) * 0.5)  # Top 50%
        threshold = sorted_conf.iloc[threshold_idx]
        
        return threshold


if __name__ == '__main__':
    print("XGBoost Trainer Test")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    # Create synthetic features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
    })
    
    # Create target (correlated with features)
    y = (X['feature_1'] + X['feature_2'] * 0.5 > 0).astype(int)
    
    # Split
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train
    trainer = XGBoostTrainer()
    
    # Cross-validation
    cv_results = trainer.cross_validate(X_train, y_train, n_splits=3)
    print(f"\nCV Accuracy: {cv_results['accuracy_mean']:.3f}")
    
    # Train final model
    trainer.train(X_train, y_train)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {metrics['accuracy']:.3f}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    print(f"\nFeature Importance:")
    print(importance)
    
    # Save and load
    model_path = trainer.save_model('test_model.pkl')
    
    # Inference
    inference = ModelInference(model_path)
    pred, prob = inference.predict(X_test.iloc[[0]])
    print(f"\nPrediction: {pred}, Probability: {prob:.3f}")
