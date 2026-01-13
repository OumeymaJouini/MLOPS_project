"""
Utility Functions
=================
Helper functions for the MLOps project.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save dictionary to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional file to write logs
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_project_root() -> str:
    """Get the project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current))


def ensure_dirs(*paths):
    """Ensure directories exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as readable string."""
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{name}: {value:.4f}")
        else:
            lines.append(f"{name}: {value}")
    return "\n".join(lines)
