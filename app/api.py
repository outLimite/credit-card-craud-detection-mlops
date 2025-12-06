from fastapi import FastAPI
import pandas as pd
from app.schemas import PredictRequest, PredictResponse
from src.inference.predict import predict_inference
from src.inference.model_loader import load_model
from app.config import MODEL_NAME

app = FastAPI(title="Fraud Detection API", version="1.0")

model = load_model(MODEL_NAME)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    df = pd.DataFrame([request.dict()])
    y_pred, y_proba = predict_inference(df, model_name=MODEL_NAME)
    return PredictResponse(
        prediction=int(y_pred[0]),
        probability=float(y_proba[0])
    )
