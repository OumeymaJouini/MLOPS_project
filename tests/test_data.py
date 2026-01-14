"""
Tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.data_loader import load_california_housing
from src.data.preprocessing import DataPreprocessor


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_california_housing(self):
        """Test that data loads correctly."""
        df = load_california_housing()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'MedHouseVal' in df.columns  # Target column name
        
    def test_data_shape(self):
        """Test expected data shape."""
        df = load_california_housing()
        
        # California Housing has 8 features + 1 target
        assert df.shape[1] == 9
        assert df.shape[0] > 20000  # ~20640 samples
        
    def test_no_missing_values(self):
        """Test that loaded data has no missing values."""
        df = load_california_housing()
        
        assert df.isnull().sum().sum() == 0


class TestPreprocessor:
    """Test preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'MedInc': [3.5, 4.2, 5.1, 6.0, 7.2],
            'HouseAge': [20, 25, 30, 35, 40],
            'AveRooms': [5.0, 6.0, 7.0, 8.0, 9.0],
            'AveBedrms': [1.0, 1.2, 1.4, 1.6, 1.8],
            'Population': [1000, 1200, 1400, 1600, 1800],
            'AveOccup': [3.0, 3.2, 3.4, 3.6, 3.8],
            'Latitude': [34.0, 34.5, 35.0, 35.5, 36.0],
            'Longitude': [-118.0, -118.5, -119.0, -119.5, -120.0],
            'MedHouseVal': [2.5, 3.0, 3.5, 4.0, 4.5]
        })
    
    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor is not None
        assert preprocessor.scaler is not None
        assert preprocessor.is_fitted == False
        
    def test_feature_creation(self, sample_data):
        """Test feature engineering creates new features."""
        preprocessor = DataPreprocessor()
        
        df_features = preprocessor.create_features(sample_data)
        
        # Should have additional engineered features
        assert 'RoomsPerPerson' in df_features.columns
        assert 'BedroomRatio' in df_features.columns
        assert 'PopulationPerHousehold' in df_features.columns
        
    def test_split_data(self, sample_data):
        """Test that data splitting works correctly."""
        preprocessor = DataPreprocessor()
        
        df_features = preprocessor.create_features(sample_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            df_features, 
            target_col='MedHouseVal',
            test_size=0.4
        )
        
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) + len(y_test) == len(sample_data)
        
    def test_fit_transform(self, sample_data):
        """Test that fit_transform scales features."""
        preprocessor = DataPreprocessor()
        
        df_features = preprocessor.create_features(sample_data)
        X = df_features.drop('MedHouseVal', axis=1)
        
        X_scaled = preprocessor.fit_transform(X)
        
        assert preprocessor.is_fitted == True
        assert X_scaled.shape == X.shape
