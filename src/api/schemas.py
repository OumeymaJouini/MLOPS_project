"""
API Request/Response Schemas
============================
Pydantic models for API validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class HousingFeatures(BaseModel):
    """Input features for housing price prediction."""
    
    MedInc: float = Field(..., description="Median income in block group", ge=0)
    HouseAge: float = Field(..., description="Median house age in block group", ge=0)
    AveRooms: float = Field(..., description="Average number of rooms per household", ge=0)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", ge=0)
    Population: float = Field(..., description="Block group population", ge=0)
    AveOccup: float = Field(..., description="Average number of household members", ge=0)
    Latitude: float = Field(..., description="Block group latitude", ge=32, le=42)
    Longitude: float = Field(..., description="Block group longitude", ge=-125, le=-114)
    
    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


class PredictionResponse(BaseModel):
    """Response with predicted house value."""
    
    predicted_value: float = Field(..., description="Predicted median house value (in $100,000s)")
    predicted_value_dollars: float = Field(..., description="Predicted value in dollars")
    model_version: str = Field(..., description="Version of the model used")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_value": 4.526,
                "predicted_value_dollars": 452600.0,
                "model_version": "v1"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    instances: List[HousingFeatures] = Field(..., description="List of housing features")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Current model version")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    model_version: str
    features: List[str]
    metrics: Optional[dict] = None
