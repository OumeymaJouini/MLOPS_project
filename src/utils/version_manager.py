"""
Model Version Management
========================
Manage model versions with rollback capability.
"""

import os
import sys
import json
import shutil
import joblib
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
MODEL_DIR = "models"
VERSION_FILE = "models/versions.json"


class ModelVersionManager:
    """Manage model versions with deployment and rollback."""
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.version_file = os.path.join(model_dir, "versions.json")
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict:
        """Load version history from file."""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {
            "current": None,
            "history": [],
            "models": {}
        }
    
    def _save_versions(self):
        """Save version history to file."""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        version: str,
        metrics: Dict = None,
        description: str = None
    ) -> bool:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file
            version: Version string (e.g., "v1", "v2")
            metrics: Model metrics
            description: Optional description
            
        Returns:
            True if successful
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Copy model to versioned name
        ext = os.path.splitext(model_path)[1]
        versioned_path = os.path.join(self.model_dir, f"model_{version}{ext}")
        
        if model_path != versioned_path:
            shutil.copy2(model_path, versioned_path)
        
        # Register version
        self.versions["models"][version] = {
            "path": versioned_path,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "description": description or ""
        }
        
        self._save_versions()
        logger.info(f"Registered model version: {version}")
        
        return True
    
    def deploy(self, version: str) -> bool:
        """
        Deploy a specific model version.
        
        Args:
            version: Version to deploy
            
        Returns:
            True if successful
        """
        if version not in self.versions["models"]:
            logger.error(f"Version not found: {version}")
            return False
        
        # Add current to history if exists
        if self.versions["current"]:
            self.versions["history"].append({
                "version": self.versions["current"],
                "deployed_until": datetime.now().isoformat()
            })
        
        # Set new current version
        self.versions["current"] = version
        
        # Create symlink or copy to "current" model
        model_info = self.versions["models"][version]
        current_path = os.path.join(self.model_dir, "model_current.joblib")
        
        if os.path.exists(current_path):
            os.remove(current_path)
        shutil.copy2(model_info["path"], current_path)
        
        self._save_versions()
        logger.info(f"Deployed version: {version}")
        
        return True
    
    def rollback(self, steps: int = 1) -> Optional[str]:
        """
        Rollback to a previous version.
        
        Args:
            steps: Number of versions to roll back
            
        Returns:
            Rolled back version or None
        """
        if len(self.versions["history"]) < steps:
            logger.error(f"Not enough history to rollback {steps} steps")
            return None
        
        # Get previous version
        prev = self.versions["history"][-steps]
        prev_version = prev["version"]
        
        # Deploy previous version
        if self.deploy(prev_version):
            # Remove from history (it's now current)
            self.versions["history"] = self.versions["history"][:-steps]
            self._save_versions()
            logger.info(f"Rolled back to: {prev_version}")
            return prev_version
        
        return None
    
    def list_versions(self) -> List[Dict]:
        """List all registered versions."""
        versions = []
        for version, info in self.versions["models"].items():
            versions.append({
                "version": version,
                "is_current": version == self.versions["current"],
                "registered_at": info["registered_at"],
                "metrics": info.get("metrics", {}),
                "description": info.get("description", "")
            })
        return sorted(versions, key=lambda x: x["registered_at"], reverse=True)
    
    def get_current(self) -> Optional[str]:
        """Get current deployed version."""
        return self.versions["current"]
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get info about a specific version."""
        return self.versions["models"].get(version)


def setup_versions():
    """Initialize version management with existing models."""
    manager = ModelVersionManager()
    
    # Find existing models
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("No models directory found")
        return
    
    # Register existing models
    for f in os.listdir(model_dir):
        if f.endswith('.joblib') and 'version' not in f:
            model_path = os.path.join(model_dir, f)
            
            # Determine version from filename
            if 'v1' in f or 'vv1' in f:
                version = 'v1'
            elif 'v2' in f:
                version = 'v2'
            elif 'experiment' in f:
                version = 'v2'  # Experiments become v2
            else:
                version = 'v1'
            
            # Try to load metrics
            try:
                data = joblib.load(model_path)
                if isinstance(data, dict) and 'metrics' in data:
                    metrics = data['metrics']
                else:
                    metrics = None
            except:
                metrics = None
            
            manager.register_model(
                model_path=model_path,
                version=version,
                metrics=metrics,
                description=f"Auto-registered from {f}"
            )
    
    # Deploy v1 as current if exists
    if 'v1' in manager.versions["models"]:
        manager.deploy('v1')
    elif manager.versions["models"]:
        # Deploy first available version
        first_version = list(manager.versions["models"].keys())[0]
        manager.deploy(first_version)
    
    print("\nVersion Management Setup Complete!")
    print("=" * 50)
    for v in manager.list_versions():
        current = " [CURRENT]" if v["is_current"] else ""
        print(f"  {v['version']}{current}: {v.get('description', '')}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Version Management")
    parser.add_argument("command", choices=["setup", "list", "deploy", "rollback"],
                       help="Command to execute")
    parser.add_argument("--version", type=str, help="Version to deploy")
    parser.add_argument("--steps", type=int, default=1, help="Rollback steps")
    
    args = parser.parse_args()
    
    manager = ModelVersionManager()
    
    if args.command == "setup":
        setup_versions()
    
    elif args.command == "list":
        print("\nRegistered Model Versions:")
        print("=" * 50)
        for v in manager.list_versions():
            current = " [CURRENT]" if v["is_current"] else ""
            metrics = v.get("metrics", {})
            r2 = metrics.get("r2", metrics.get("test_r2", "N/A"))
            print(f"  {v['version']}{current}")
            print(f"    R²: {r2}")
            print(f"    Registered: {v['registered_at']}")
        print("=" * 50)
    
    elif args.command == "deploy":
        if not args.version:
            print("Error: --version required for deploy")
        else:
            manager.deploy(args.version)
    
    elif args.command == "rollback":
        manager.rollback(args.steps)
