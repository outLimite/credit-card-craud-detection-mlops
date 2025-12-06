from catboost import CatBoostClassifier
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def load_model(model_name="best_catboost_fraud_model.cbm") -> CatBoostClassifier:
    model_path = MODELS_DIR / model_name
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model
