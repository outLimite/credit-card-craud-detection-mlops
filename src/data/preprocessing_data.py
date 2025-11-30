import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


def train_val_test_split_df(DATA_PATH: str = "src/data/dataset/creditcard.csv", RANDOM_STATE: int = 42, scale_features: bool = False):
    """Разделяет данные на обучающую, валидационную и тестовую выборки с масштабированием."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.3, random_state=RANDOM_STATE, stratify=y_test
    )

    scaler = None
    if scale_features:
        scaler = RobustScaler()
        
        X_train_scaled = X_train.copy()
        X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
        
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_val_scaled[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])
        X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def apply_sampling(X, y, method: str = "smote", sampling_strategy: str = "auto", RANDOM_STATE: int = 42):
    """Применяет методы сэмплинга для балансировки классов."""
    if method == 'smote':
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=5,
            random_state=RANDOM_STATE
        )
    elif method == 'adasyn':
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=5,
            random_state=RANDOM_STATE
        )
    elif method == 'undersample':
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, 
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError("Доступные методы: 'smote', 'adasyn', 'undersample'")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled