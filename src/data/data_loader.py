"""
Data Loader Module
==================
Handles loading California Housing dataset from sklearn or local files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_california_housing(save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load California Housing dataset from sklearn.
    
    Args:
        save_path: Optional path to save the raw data as CSV
        
    Returns:
        DataFrame with features and target
    """
    logger.info("Loading California Housing dataset...")
    
    # Fetch from sklearn
    housing = fetch_california_housing(as_frame=True)
    
    # Combine features and target
    df = housing.frame
    
    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    logger.info(f"Features: {list(housing.feature_names)}")
    logger.info(f"Target: MedHouseVal (Median House Value in $100k)")
    
    # Save to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
    
    return df


def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with the data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {file_path}: {df.shape}")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "features": list(df.columns[:-1]),
        "target": df.columns[-1],
        "missing_values": df.isnull().sum().to_dict(),
        "target_stats": {
            "mean": df.iloc[:, -1].mean(),
            "std": df.iloc[:, -1].std(),
            "min": df.iloc[:, -1].min(),
            "max": df.iloc[:, -1].max()
        }
    }
    return summary


if __name__ == "__main__":
    # Test the data loader
    df = load_california_housing(save_path="data/raw/housing.csv")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary:")
    print(get_data_summary(df))
