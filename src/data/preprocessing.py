"""
Data Preprocessing Module
=========================
Feature engineering and data transformation for California Housing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Feature engineering
        # Rooms per person
        df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']
        
        # Bedrooms ratio
        df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
        
        # Population density proxy
        df['PopulationPerHousehold'] = df['Population'] / (df['AveOccup'] + 1)
        
        # Handle infinities and NaN from division
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        logger.info(f"Created features. New shape: {df.shape}")
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'MedHouseVal',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Scaled features as numpy array
        """
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        logger.info("Scaler fitted and data transformed")
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Scaled features as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.scaler.transform(X)
    
    def save(self, path: str):
        """
        Save preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """
        Load preprocessor from disk.
        
        Args:
            path: Path to load the preprocessor from
        """
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Preprocessor loaded from {path}")


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        test_size: Fraction for test set
        random_state: Random seed
        save_dir: Directory to save processed data
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = DataPreprocessor()
    
    # Feature engineering
    df_features = preprocessor.create_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_features, 
        test_size=test_size,
        random_state=random_state
    )
    
    # Scale features
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save processed data and preprocessor
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X_train.npy"), X_train_scaled)
        np.save(os.path.join(save_dir, "X_test.npy"), X_test_scaled)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train.values)
        np.save(os.path.join(save_dir, "y_test.npy"), y_test.values)
        preprocessor.save(os.path.join(save_dir, "preprocessor.joblib"))
        logger.info(f"Processed data saved to {save_dir}")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, preprocessor


if __name__ == "__main__":
    from data_loader import load_california_housing
    
    # Load and preprocess
    df = load_california_housing()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)
    
    print(f"\nProcessed shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
