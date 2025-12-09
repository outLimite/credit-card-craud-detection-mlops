import os
import sys
from pathlib import Path

import numpy as np
import mlflow
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data.preprocessing_data import train_val_test_split_df, apply_sampling

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "data/plots"
CATBOOST_INFO_DIR = BASE_DIR / "catboost_info"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CATBOOST_INFO_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://localhost:5050"
EXPERIMENT_NAME = "catboost-fraud-classification-optuna"

mlflow.set_tracking_uri(MLFLOW_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    mlflow.set_experiment(experiment.experiment_id)
else:
    mlflow.create_experiment(EXPERIMENT_NAME)

def get_model(params):
    """Создает модель CatBoost для классификации"""
    model = CatBoostClassifier(
        verbose=0,
        train_dir="./catboost_info",
        random_state=42,
        **params,
    )
    return model

def run_optimization(num_trials: int):
    X_train, X_val, X_test, y_train, y_val, y_test, _ = train_val_test_split_df()
    
    X_train_sampled, y_train_sampled = apply_sampling(X_train, y_train)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }
        
        scale_pos_weight = len(y_train_sampled[y_train_sampled == 0]) / len(y_train_sampled[y_train_sampled == 1])
        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 
                                                        max(0.1, scale_pos_weight * 0.5), 
                                                        min(10.0, scale_pos_weight * 2.0))
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            
            model = get_model(params)
            model.fit(
                X_train_sampled, y_train_sampled,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("trial_number", trial.number)
            
            trial.set_user_attr("model_type", "CatBoostClassifier")
            trial.set_user_attr("best_roc_auc", roc_auc)
            trial.set_user_attr("best_f1", f1)
        
        return roc_auc 

    study = optuna.create_study(
        direction='maximize',
        study_name='catboost_fraud_classification',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    mlflow_callback = optuna.integration.MLflowCallback(
        tracking_uri=MLFLOW_URI,
        metric_name="roc_auc",
    )
    
    study.optimize(
        objective, 
        n_trials=num_trials,
        callbacks=[mlflow_callback],
        gc_after_trial=True
    )
    
    with mlflow.start_run(run_name="best_params"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)
        mlflow.log_metric("best_trial", study.best_trial.number)
        
        best_model = get_model(study.best_params)
        best_model.fit(
            pd.concat([X_train_sampled, X_val]), 
            pd.concat([y_train_sampled, y_val])
        )
        
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = best_model.predict(X_test)
        
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_f1 = f1_score(y_test, y_test_pred)
        
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_f1_score", test_f1)
        
        best_model_path = MODELS_DIR / "best_catboost_fraud_model.cbm"
        best_model.save_model(str(best_model_path))
        mlflow.log_artifact(str(best_model_path), artifact_path="best_model")
    
    print("\nResults")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    print(f"  Value (ROC-AUC): {study.best_value:.4f}")
    print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study

if __name__ == '__main__':
    study = run_optimization(num_trials=5) 