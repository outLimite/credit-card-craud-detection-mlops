from catboost import CatBoostClassifier
import pandas as pd
from .model_loader import load_model 

def predict_inference(X: pd.DataFrame, model_name: str = "best_catboost_fraud_model.cbm"):
    model: CatBoostClassifier = load_model(model_name=model_name)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return y_pred, y_proba
