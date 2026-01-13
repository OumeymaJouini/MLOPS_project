"""
FastAPI Inference Service
=========================
REST API for California Housing price predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.schemas import (
    HousingFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "data/processed/preprocessor.joblib")

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="MLOps Project - Predict median house values in California",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessor
model = None
preprocessor = None
current_version = MODEL_VERSION


def get_model_path(version: str) -> str:
    """Get model path for a specific version."""
    # Try different naming conventions
    paths = [
        os.path.join(MODEL_DIR, f"gradient_boosting_{version}.joblib"),
        os.path.join(MODEL_DIR, f"random_forest_{version}.joblib"),
        os.path.join(MODEL_DIR, f"model_{version}.joblib"),
        os.path.join(MODEL_DIR, "gradient_boosting_experiment.joblib"),
        os.path.join(MODEL_DIR, "random_forest_experiment.joblib"),
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    
    # Fallback to any joblib file
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.joblib'):
                return os.path.join(MODEL_DIR, f)
    
    return None


def load_model(version: str = None):
    """Load model and preprocessor."""
    global model, preprocessor, current_version
    
    version = version or MODEL_VERSION
    model_path = get_model_path(version)
    
    if model_path is None:
        logger.error(f"No model found for version {version}")
        return False
    
    try:
        # Load model
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
        else:
            model = model_data
        
        current_version = version
        logger.info(f"Model loaded from {model_path}")
        
        # Load preprocessor
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning("No preprocessor found, using raw features")
            preprocessor = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_model()
    if not success:
        logger.warning("Model not loaded on startup. Use /reload endpoint to load.")


def prepare_features(features: HousingFeatures) -> np.ndarray:
    """Prepare features for prediction."""
    # Create DataFrame
    df = pd.DataFrame([{
        "MedInc": features.MedInc,
        "HouseAge": features.HouseAge,
        "AveRooms": features.AveRooms,
        "AveBedrms": features.AveBedrms,
        "Population": features.Population,
        "AveOccup": features.AveOccup,
        "Latitude": features.Latitude,
        "Longitude": features.Longitude
    }])
    
    # Add engineered features (same as in preprocessing)
    df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']
    df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
    df['PopulationPerHousehold'] = df['Population'] / (df['AveOccup'] + 1)
    
    # Handle infinities
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    # Apply scaler if available
    if preprocessor is not None:
        try:
            X = preprocessor.transform(df.values)
        except Exception as e:
            logger.warning(f"Preprocessor failed, using raw features: {e}")
            X = df.values
    else:
        X = df.values
    
    return X


# ==================== API Endpoints ====================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "California Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=current_version
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
        "RoomsPerPerson", "BedroomRatio", "PopulationPerHousehold"
    ]
    
    return ModelInfo(
        model_name=type(model).__name__,
        model_version=current_version,
        features=features,
        metrics=None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: HousingFeatures):
    """
    Predict house value for given features.
    
    Returns the predicted median house value in $100,000s.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        X = prepare_features(features)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return PredictionResponse(
            predicted_value=round(float(prediction), 4),
            predicted_value_dollars=round(float(prediction) * 100000, 2),
            model_version=current_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple instances.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for features in request.instances:
        try:
            X = prepare_features(features)
            prediction = model.predict(X)[0]
            predictions.append(PredictionResponse(
                predicted_value=round(float(prediction), 4),
                predicted_value_dollars=round(float(prediction) * 100000, 2),
                model_version=current_version
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions)
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model(version: str = None):
    """Reload the model, optionally with a different version."""
    success = load_model(version)
    if success:
        return {"status": "success", "version": current_version}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


@app.get("/model/versions", tags=["Model"])
async def list_versions():
    """List available model versions."""
    versions = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.joblib'):
                versions.append(f)
    return {"available_models": versions}


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
