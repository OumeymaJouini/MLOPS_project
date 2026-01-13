"""
Model Definitions Module
========================
Scikit-learn models for California Housing prediction.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model(model_name: str, params: Dict[str, Any] = None):
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model
        params: Model hyperparameters
        
    Returns:
        Sklearn model instance
    """
    params = params or {}
    
    models = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "svr": SVR,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name](**params)
    logger.info(f"Created model: {model_name} with params: {params}")
    return model


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for each model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "random_state": 42
        },
        "ridge": {
            "alpha": 1.0,
            "random_state": 42
        },
        "lasso": {
            "alpha": 0.1,
            "random_state": 42
        },
        "elastic_net": {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "random_state": 42
        },
        "svr": {
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1
        }
    }
    return defaults.get(model_name, {})


def save_model(model, path: str, metadata: Dict[str, Any] = None):
    """
    Save model to disk with optional metadata.
    
    Args:
        model: Trained sklearn model
        path: Path to save the model
        metadata: Optional metadata dict
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "model": model,
        "metadata": metadata or {}
    }
    
    joblib.dump(save_dict, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    """
    Load model from disk.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    save_dict = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return save_dict["model"], save_dict.get("metadata", {})


class HousingModel:
    """Wrapper class for housing price prediction model."""
    
    def __init__(self, model_name: str = "random_forest", params: Dict[str, Any] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model to use
            params: Model hyperparameters
        """
        self.model_name = model_name
        self.params = params or get_default_params(model_name)
        self.model = get_model(model_name, self.params)
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Training complete!")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def save(self, path: str, metadata: Dict[str, Any] = None):
        """Save model to disk."""
        full_metadata = {
            "model_name": self.model_name,
            "params": self.params,
            **(metadata or {})
        }
        save_model(self.model, path, full_metadata)
        
    def load(self, path: str):
        """Load model from disk."""
        self.model, metadata = load_model(path)
        self.model_name = metadata.get("model_name", "unknown")
        self.params = metadata.get("params", {})
        self.is_trained = True
        return metadata


if __name__ == "__main__":
    # Test model creation
    print("Available models:")
    for name in ["random_forest", "gradient_boosting", "ridge", "lasso"]:
        model = get_model(name, get_default_params(name))
        print(f"  - {name}: {type(model).__name__}")
