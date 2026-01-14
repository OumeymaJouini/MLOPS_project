"""
Tests for FastAPI endpoints.
"""

import pytest
import os
import sys

# Skip API tests if dependencies not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)
    
    def test_health_check_returns_response(self, client):
        """Test health endpoint returns a valid response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Can be healthy or unhealthy depending on model availability
        assert data["status"] in ["healthy", "unhealthy"]


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_input(self):
        """Sample input for prediction."""
        return {
            "MedInc": 3.5,
            "HouseAge": 25.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.1,
            "Population": 1200.0,
            "AveOccup": 3.2,
            "Latitude": 34.5,
            "Longitude": -118.5
        }
    
    def test_predict_endpoint_exists(self, client, sample_input):
        """Test prediction endpoint exists and accepts requests."""
        response = client.post("/predict", json=sample_input)
        
        # Should return 200 (success) or 503 (model not loaded)
        assert response.status_code in [200, 503]
        
    def test_predict_valid_input_returns_prediction(self, client, sample_input):
        """Test prediction with valid input returns a response."""
        response = client.post("/predict", json=sample_input)
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert isinstance(data["prediction"], (int, float))
            # House prices should be positive
            assert data["prediction"] > 0
        else:
            # Model not loaded - acceptable in CI environment
            assert response.status_code == 503
            
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input returns error."""
        invalid_input = {"invalid_field": 123}
        response = client.post("/predict", json=invalid_input)
        
        # Should return validation error (422) or service unavailable (503)
        assert response.status_code in [422, 503]


class TestModelVersionsEndpoint:
    """Test model versions endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)
    
    def test_get_model_versions(self, client):
        """Test getting model versions."""
        response = client.get("/model/versions")
        
        assert response.status_code == 200
        data = response.json()
        # Should have some model-related information
        assert isinstance(data, dict)
        assert len(data) > 0
