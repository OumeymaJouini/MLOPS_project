"""
Training Module with MLflow Integration
=======================================
Main training script for California Housing prediction.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import load_california_housing, load_from_csv
from src.data.preprocessing import preprocess_pipeline, DataPreprocessor
from src.models.model import HousingModel, get_default_params
from src.evaluation.metrics import calculate_metrics, evaluate_model

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    model_name: str = "random_forest",
    params: Dict[str, Any] = None,
    config_path: str = "configs/config.yaml",
    use_mlflow: bool = True,
    experiment_name: str = None
) -> Dict[str, Any]:
    """
    Train a model with optional MLflow tracking.
    
    Args:
        model_name: Name of the model to train
        params: Model hyperparameters (uses defaults if None)
        config_path: Path to config file
        use_mlflow: Whether to log to MLflow
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with model, metrics, and paths
    """
    # Load config
    config = load_config(config_path)
    
    # Get parameters
    if params is None:
        params = get_default_params(model_name)
    
    logger.info(f"Starting training: {model_name}")
    logger.info(f"Parameters: {params}")
    
    # Setup MLflow
    if use_mlflow:
        # Use local tracking (file-based, no server needed)
        mlflow.set_tracking_uri("file:./mlruns")
        exp_name = experiment_name or config.get('mlflow', {}).get('experiment_name', 'california-housing')
        mlflow.set_experiment(exp_name)
    
    # Load data
    raw_data_path = config['data']['raw_path']
    if os.path.exists(raw_data_path):
        logger.info(f"Loading data from {raw_data_path}")
        df = load_from_csv(raw_data_path)
    else:
        logger.info("Downloading California Housing dataset...")
        df = load_california_housing(save_path=raw_data_path)
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        save_dir=config['data']['processed_path']
    )
    
    # Create and train model
    model = HousingModel(model_name, params)
    
    # MLflow tracking
    if use_mlflow:
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Train
            model.train(X_train, y_train)
            
            # Evaluate
            train_metrics = calculate_metrics(y_train, model.predict(X_train))
            test_metrics = calculate_metrics(y_test, model.predict(X_test))
            
            # Log metrics
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)
            
            # Log model
            mlflow.sklearn.log_model(model.model, "model")
            
            # Save model locally
            model_path = f"models/{model_name}_v{config['model']['version']}.joblib"
            model.save(model_path, metadata={
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "timestamp": datetime.now().isoformat()
            })
            mlflow.log_artifact(model_path)
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
    else:
        # Train without MLflow
        model.train(X_train, y_train)
        test_metrics = calculate_metrics(y_test, model.predict(X_test))
        train_metrics = calculate_metrics(y_train, model.predict(X_train))
        
        model_path = f"models/{model_name}.joblib"
        model.save(model_path)
        run_id = None
    
    # Print results
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE: {model_name}")
    print("="*50)
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print("="*50)
    
    return {
        "model": model,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "model_path": model_path,
        "run_id": run_id
    }


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train California Housing model")
    parser.add_argument("--model", type=str, default="random_forest",
                       choices=["random_forest", "gradient_boosting", "ridge", "lasso", "elastic_net"],
                       help="Model to train")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow tracking")
    parser.add_argument("--experiment", type=str, default=None,
                       help="MLflow experiment name")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        config_path=args.config,
        use_mlflow=not args.no_mlflow,
        experiment_name=args.experiment
    )


if __name__ == "__main__":
    main()
