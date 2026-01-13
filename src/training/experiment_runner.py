"""
Experiment Runner
=================
Run multiple model experiments and compare results.
"""

import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, List, Any
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import load_from_csv, load_california_housing
from src.data.preprocessing import preprocess_pipeline
from src.models.model import get_model, get_default_params
from src.evaluation.metrics import calculate_metrics, compare_models

import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run multiple model experiments with MLflow tracking."""
    
    def __init__(
        self,
        experiment_name: str = "california-housing-experiment",
        data_path: str = "data/raw/housing.csv"
    ):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        
    def load_data(self):
        """Load and preprocess data."""
        logger.info("Loading data...")
        
        if os.path.exists(self.data_path):
            df = load_from_csv(self.data_path)
        else:
            df = load_california_housing(save_path=self.data_path)
        
        self.X_train, self.X_test, self.y_train, self.y_test, _ = preprocess_pipeline(df)
        
        logger.info(f"Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")
        
    def run_experiment(
        self,
        model_name: str,
        params: Dict[str, Any] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single model experiment.
        
        Args:
            model_name: Name of the model
            params: Model parameters (uses defaults if None)
            save_model: Whether to save the model locally
            
        Returns:
            Dictionary with metrics and run info
        """
        if self.X_train is None:
            self.load_data()
        
        # Get parameters
        if params is None:
            params = get_default_params(model_name)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment: {model_name}")
        logger.info(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            
            # Create and train model
            model = get_model(model_name, params)
            
            start_time = datetime.now()
            model.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            mlflow.log_metric("training_time_seconds", training_time)
            
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
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            if save_model:
                model_path = f"models/{model_name}_experiment.joblib"
                os.makedirs("models", exist_ok=True)
                joblib.dump({
                    "model": model,
                    "params": params,
                    "metrics": test_metrics,
                    "timestamp": datetime.now().isoformat()
                }, model_path)
                mlflow.log_artifact(model_path)
            
            run_id = mlflow.active_run().info.run_id
            
        # Store results
        self.results[model_name] = {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "params": params,
            "run_id": run_id,
            "training_time": training_time
        }
        
        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Test R²:   {test_metrics['r2']:.4f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")
        print(f"  Time:      {training_time:.2f}s")
        
        return self.results[model_name]
    
    def run_all_experiments(
        self,
        models: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run experiments for multiple models.
        
        Args:
            models: List of model names (uses defaults if None)
            
        Returns:
            Dictionary with all results
        """
        if models is None:
            models = ["random_forest", "gradient_boosting", "ridge", "elastic_net"]
        
        logger.info(f"Running {len(models)} experiments...")
        
        for model_name in models:
            try:
                self.run_experiment(model_name)
            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                continue
        
        return self.results
    
    def compare_results(self) -> str:
        """Compare all experiment results and return best model."""
        if not self.results:
            logger.warning("No results to compare. Run experiments first.")
            return None
        
        # Format for comparison
        metrics_dict = {
            name: result["test_metrics"]
            for name, result in self.results.items()
        }
        
        best_model = compare_models(metrics_dict)
        
        return best_model
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get the best model based on R² score."""
        if not self.results:
            return None
        
        best_name = None
        best_r2 = -np.inf
        
        for name, result in self.results.items():
            r2 = result["test_metrics"]["r2"]
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        
        return {
            "model_name": best_name,
            "r2": best_r2,
            "run_id": self.results[best_name]["run_id"],
            "params": self.results[best_name]["params"]
        }


def run_all_experiments():
    """Main function to run all experiments."""
    
    print("\n" + "="*60)
    print("MLOPS PROJECT - DAY 2: MODEL EXPERIMENTS")
    print("="*60 + "\n")
    
    runner = ExperimentRunner()
    runner.load_data()
    
    # Run all model experiments
    models = ["random_forest", "gradient_boosting", "ridge", "elastic_net"]
    runner.run_all_experiments(models)
    
    # Compare results
    print("\n")
    best_model = runner.compare_results()
    
    # Get best model info
    best_info = runner.get_best_model()
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_info['model_name']}")
    print(f"   R² Score: {best_info['r2']:.4f}")
    print(f"   MLflow Run ID: {best_info['run_id']}")
    print(f"{'='*60}\n")
    
    print("View all experiments in MLflow UI: http://localhost:5000")
    
    return runner.results


if __name__ == "__main__":
    results = run_all_experiments()
