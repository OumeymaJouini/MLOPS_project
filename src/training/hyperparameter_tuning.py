"""
Hyperparameter Tuning with Optuna
=================================
Automated hyperparameter optimization for housing price models.
"""

import os
import sys
import optuna
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import load_from_csv, load_california_housing
from src.data.preprocessing import preprocess_pipeline
from src.models.model import get_model
from src.evaluation.metrics import calculate_metrics

import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning with MLflow tracking."""
    
    def __init__(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_trials: int = 20,
        cv_folds: int = 3,
        experiment_name: str = "hyperparameter-tuning"
    ):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.experiment_name = experiment_name
        self.best_params = None
        self.best_score = None
        
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space for each model."""
        
        if self.model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
                "n_jobs": -1
            }
            
        elif self.model_name == "gradient_boosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42
            }
            
        elif self.model_name == "ridge":
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 100, log=True),
            }
            
        elif self.model_name == "lasso":
            return {
                "alpha": trial.suggest_float("alpha", 0.0001, 10, log=True),
            }
            
        elif self.model_name == "elastic_net":
            return {
                "alpha": trial.suggest_float("alpha", 0.0001, 10, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            }
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Get hyperparameters for this trial
        params = self._get_param_space(trial)
        
        # Create model
        model = get_model(self.model_name, params)
        
        # Cross-validation score (negative MSE -> we want to maximize)
        # Note: n_jobs=1 to avoid Windows multiprocessing issues with Python 3.7
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        # Return mean CV score (Optuna minimizes by default, but we use neg_mse)
        rmse = np.sqrt(-scores.mean())
        
        return rmse
    
    def tune(self, log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            log_to_mlflow: Whether to log results to MLflow
            
        Returns:
            Dictionary with best parameters and score
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_name}")
        logger.info(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
        
        # Create Optuna study (minimize RMSE)
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{self.model_name}_tuning"
        )
        
        # Optimize
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best RMSE: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Train final model with best params and log to MLflow
        if log_to_mlflow:
            self._log_best_model_to_mlflow()
        
        return {
            "best_params": self.best_params,
            "best_rmse": self.best_score,
            "n_trials": self.n_trials,
            "model_name": self.model_name
        }
    
    def _log_best_model_to_mlflow(self):
        """Train and log the best model to MLflow."""
        
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"{self.model_name}_tuned_{datetime.now().strftime('%H%M%S')}"):
            # Add fixed params back
            full_params = self.best_params.copy()
            if self.model_name in ["random_forest", "gradient_boosting"]:
                full_params["random_state"] = 42
            if self.model_name == "random_forest":
                full_params["n_jobs"] = -1
            
            # Log parameters
            mlflow.log_params(full_params)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("tuning_trials", self.n_trials)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("is_tuned", True)
            
            # Train model with best params
            model = get_model(self.model_name, full_params)
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            train_metrics = calculate_metrics(self.y_train, train_pred)
            test_metrics = calculate_metrics(self.y_test, test_pred)
            
            # Log metrics
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Logged tuned model to MLflow")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")


def run_tuning(
    model_name: str = "random_forest",
    n_trials: int = 20,
    data_path: str = "data/raw/housing.csv"
) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter tuning.
    
    Args:
        model_name: Name of model to tune
        n_trials: Number of Optuna trials
        data_path: Path to data file
        
    Returns:
        Tuning results
    """
    # Load data
    if os.path.exists(data_path):
        df = load_from_csv(data_path)
    else:
        df = load_california_housing(save_path=data_path)
    
    # Preprocess
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(df)
    
    # Create tuner
    tuner = HyperparameterTuner(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=n_trials,
        experiment_name="california-housing-experiment"
    )
    
    # Run tuning
    results = tuner.tune(log_to_mlflow=True)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--model", type=str, default="random_forest", 
                       choices=["random_forest", "gradient_boosting", "ridge", "lasso", "elastic_net"])
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: {args.model.upper()}")
    print(f"{'='*60}\n")
    
    results = run_tuning(model_name=args.model, n_trials=args.trials)
    
    print(f"\n{'='*60}")
    print("TUNING COMPLETE!")
    print(f"Best RMSE: {results['best_rmse']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"{'='*60}\n")
