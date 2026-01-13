"""
Evaluation Metrics Module
=========================
Metrics for evaluating housing price predictions.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    
    # Log metrics
    logger.info("Evaluation Metrics:")
    logger.info(f"  MSE:  {metrics['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)


def compare_models(results: Dict[str, Dict[str, float]]) -> str:
    """
    Compare multiple models based on their metrics.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        
    Returns:
        Name of the best model (by R²)
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*60)
    
    best_model = None
    best_r2 = -np.inf
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} {metrics['r2']:<12.4f}")
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model_name
    
    print("-"*60)
    print(f"Best model: {best_model} (R² = {best_r2:.4f})")
    print("="*60)
    
    return best_model


if __name__ == "__main__":
    # Test metrics calculation
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("\nTest metrics:", metrics)
