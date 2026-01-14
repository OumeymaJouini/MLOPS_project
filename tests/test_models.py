"""
Tests for model functionality.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from src.models.model import get_model, get_default_params


class TestModelFactory:
    """Test model factory function."""
    
    def test_get_random_forest(self):
        """Test random forest model creation."""
        model = get_model("random_forest")
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
    def test_get_gradient_boosting(self):
        """Test gradient boosting model creation."""
        model = get_model("gradient_boosting")
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
    def test_get_ridge(self):
        """Test ridge regression model creation."""
        model = get_model("ridge")
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model("invalid_model")
            
    def test_model_with_params(self):
        """Test model creation with custom parameters."""
        params = {"n_estimators": 50, "max_depth": 5}
        model = get_model("random_forest", params)
        
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_random_forest_training(self, sample_regression_data):
        """Test random forest can be trained."""
        X, y = sample_regression_data
        
        model = get_model("random_forest", {"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
        
    def test_gradient_boosting_training(self, sample_regression_data):
        """Test gradient boosting can be trained."""
        X, y = sample_regression_data
        
        model = get_model("gradient_boosting", {"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        
    def test_model_score(self, sample_regression_data):
        """Test that trained model achieves reasonable score."""
        X, y = sample_regression_data
        
        model = get_model("random_forest", {"n_estimators": 50, "random_state": 42})
        model.fit(X, y)
        
        score = model.score(X, y)
        
        # Should achieve high score on training data
        assert score > 0.9


class TestDefaultParams:
    """Test default parameters function."""
    
    def test_random_forest_defaults(self):
        """Test random forest default parameters."""
        params = get_default_params("random_forest")
        
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "random_state" in params
        
    def test_gradient_boosting_defaults(self):
        """Test gradient boosting default parameters."""
        params = get_default_params("gradient_boosting")
        
        assert isinstance(params, dict)
        assert "n_estimators" in params
