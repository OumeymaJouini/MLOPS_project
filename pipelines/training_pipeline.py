"""
ML Training Pipeline
====================
A simplified pipeline implementation for ML training workflow.
Compatible with Python 3.7+ (ZenML requires Python 3.8+)

This pipeline orchestrates the complete ML workflow:
1. Data Loading
2. Preprocessing
3. Training
4. Evaluation
5. Model Registration
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Callable
from functools import wraps

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import joblib

from src.data.data_loader import load_california_housing, load_from_csv
from src.data.preprocessing import preprocess_pipeline
from src.models.model import get_model, get_default_params
from src.evaluation.metrics import calculate_metrics

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Pipeline Decorators ====================

def step(name: str = None):
    """Decorator to mark a function as a pipeline step."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            step_name = name or func.__name__
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(f"✓ {step_name} completed in {duration:.2f}s")
            return result
        
        wrapper._is_step = True
        wrapper._step_name = name or func.__name__
        return wrapper
    return decorator


class Pipeline:
    """Simple pipeline orchestrator."""
    
    def __init__(self, name: str, steps: list = None):
        self.name = name
        self.steps = steps or []
        self.artifacts = {}
        self.run_id = None
        
    def add_step(self, step_func: Callable):
        """Add a step to the pipeline."""
        self.steps.append(step_func)
        return self
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute all pipeline steps."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# PIPELINE: {self.name}")
        logger.info(f"# Started: {datetime.now().isoformat()}")
        logger.info(f"{'#'*60}\n")
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.artifacts = {"run_id": self.run_id, **kwargs}
        
        start_time = time.time()
        
        for step_func in self.steps:
            try:
                result = step_func(self.artifacts)
                if result:
                    self.artifacts.update(result)
            except Exception as e:
                logger.error(f"Pipeline failed at step {step_func.__name__}: {e}")
                raise
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"# PIPELINE COMPLETE")
        logger.info(f"# Total time: {total_time:.2f}s")
        logger.info(f"{'#'*60}\n")
        
        return self.artifacts


# ==================== Pipeline Steps ====================

@step("Load Data")
def load_data_step(artifacts: Dict) -> Dict:
    """Load the California Housing dataset."""
    data_path = artifacts.get("data_path", "data/raw/housing.csv")
    
    if os.path.exists(data_path):
        df = load_from_csv(data_path)
    else:
        df = load_california_housing(save_path=data_path)
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return {"dataframe": df, "n_samples": len(df)}


@step("Preprocess Data")
def preprocess_step(artifacts: Dict) -> Dict:
    """Preprocess and split the data."""
    df = artifacts["dataframe"]
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        df,
        test_size=0.2,
        random_state=42,
        save_dir="data/processed"
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor
    }


@step("Train Model")
def train_step(artifacts: Dict) -> Dict:
    """Train the model with MLflow tracking."""
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    model_name = artifacts.get("model_name", "random_forest")
    params = artifacts.get("params", get_default_params(model_name))
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("pipeline-runs")
    
    with mlflow.start_run(run_name=f"pipeline_{artifacts['run_id']}"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", len(X_train))
        
        # Train
        model = get_model(model_name, params)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
    
    logger.info(f"Model trained: {model_name}")
    logger.info(f"MLflow run ID: {run_id}")
    
    return {"model": model, "mlflow_run_id": run_id}


@step("Evaluate Model")
def evaluate_step(artifacts: Dict) -> Dict:
    """Evaluate the model on test data."""
    model = artifacts["model"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Log to MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_id=artifacts["mlflow_run_id"]):
        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)
    
    logger.info(f"Test R²: {metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    
    return {"metrics": metrics}


@step("Register Model")
def register_step(artifacts: Dict) -> Dict:
    """Save and register the trained model."""
    model = artifacts["model"]
    metrics = artifacts["metrics"]
    model_name = artifacts.get("model_name", "random_forest")
    
    # Save model
    model_path = f"models/{model_name}_pipeline.joblib"
    os.makedirs("models", exist_ok=True)
    
    joblib.dump({
        "model": model,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    return {"model_path": model_path}


# ==================== Main Pipeline ====================

def create_training_pipeline() -> Pipeline:
    """Create the complete training pipeline."""
    pipeline = Pipeline("california-housing-training")
    
    pipeline.add_step(load_data_step)
    pipeline.add_step(preprocess_step)
    pipeline.add_step(train_step)
    pipeline.add_step(evaluate_step)
    pipeline.add_step(register_step)
    
    return pipeline


def run_pipeline(model_name: str = "random_forest", params: Dict = None):
    """Run the complete training pipeline."""
    pipeline = create_training_pipeline()
    
    artifacts = pipeline.run(
        model_name=model_name,
        params=params or get_default_params(model_name)
    )
    
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Test R²: {artifacts['metrics']['r2']:.4f}")
    print(f"Test RMSE: {artifacts['metrics']['rmse']:.4f}")
    print(f"Model saved: {artifacts['model_path']}")
    print(f"MLflow Run: {artifacts['mlflow_run_id']}")
    print("="*60)
    
    return artifacts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML training pipeline")
    parser.add_argument("--model", type=str, default="random_forest",
                       choices=["random_forest", "gradient_boosting", "ridge"])
    
    args = parser.parse_args()
    
    run_pipeline(model_name=args.model)
