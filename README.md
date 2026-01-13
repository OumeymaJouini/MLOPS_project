# 🏠 California Housing Price Prediction - MLOps Project

A complete MLOps pipeline for predicting median house values in California using the California Housing dataset.

## 📋 Project Overview

This project demonstrates a production-ready MLOps workflow including:
- **Experiment Tracking**: MLflow for logging experiments
- **Hyperparameter Tuning**: Optuna for optimization
- **Model Serving**: FastAPI for inference
- **Version Management**: Custom version control with rollback
- **Containerization**: Docker & Docker Compose

## 🏆 Model Performance

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| **GradientBoosting (tuned)** | **0.8392** | **0.4591** | **0.2988** |
| GradientBoosting (default) | 0.8201 | 0.4855 | 0.3248 |
| RandomForest | 0.8012 | 0.5104 | 0.3332 |
| Ridge | 0.6525 | 0.6748 | 0.4877 |
| ElasticNet | 0.5425 | 0.7743 | 0.5747 |

## 🏗️ Project Structure

```
MLOPS_project/
├── configs/                 # Configuration files
│   └── config.yaml          # Main configuration
├── data/                    # Data directory
│   ├── raw/                 # Raw data (housing.csv)
│   └── processed/           # Preprocessed arrays & scaler
├── src/                     # Source code
│   ├── api/                 # FastAPI service
│   ├── data/                # Data loading & preprocessing
│   ├── models/              # Model definitions
│   ├── training/            # Training & tuning scripts
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Version management
├── models/                  # Saved models (.joblib)
├── mlruns/                  # MLflow experiment logs
├── docker/                  # Dockerfiles
├── docker-compose.yml       # Docker Compose setup
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd MLOPS_project
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
# or: source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python -m src.training.train
```

5. **Run experiments (multiple models)**
```bash
python -m src.training.experiment_runner
```

6. **Hyperparameter tuning**
```bash
python -m src.training.hyperparameter_tuning --model gradient_boosting --trials 10
```

### Using Docker

```bash
docker-compose up --build
```

## 📊 Dataset

**California Housing Dataset** - Median house values for California districts.

Features:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude
- **Target**: `MedHouseVal` - Median house value (in $100,000s)

## 🔧 MLOps Components

### 1. Experiment Tracking (MLflow)
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Open http://localhost:5000 to view experiments.

### 2. Hyperparameter Tuning (Optuna)
```bash
python -m src.training.hyperparameter_tuning --model gradient_boosting --trials 20
```

### 3. Model Serving (FastAPI)
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000/docs for API documentation.

### 4. Version Management
```bash
# List versions
python -m src.utils.version_manager list

# Deploy specific version
python -m src.utils.version_manager deploy --version v2

# Rollback to previous version
python -m src.utils.version_manager rollback
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model information |
| `/model/versions` | GET | List available models |
| `/model/reload` | POST | Reload model |

### Example Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23}'
```

## 📈 Model Versions

| Version | Model | Test R² | Status |
|---------|-------|---------|--------|
| v1 | RandomForest | 0.8012 | Production |
| v2 | GradientBoosting (tuned) | 0.8392 | Available |

## 📝 License

MIT License

## 👤 Author

MLOps Mini-Project - 2026
