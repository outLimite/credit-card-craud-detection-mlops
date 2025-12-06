from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAME = "best_catboost_fraud_model.cbm"
API_HOST = "0.0.0.0"
API_PORT = 8000