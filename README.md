credit-card-fraud-detection-mlops/
│
├── app/                    # API сервис
│   ├── __init__.py
│   ├── api.py              # FastAPI endpoints
│   ├── config.py           # конфиги сервиса
│   └── schemas.py          # Pydantic модели для request/response
│   
├── src/                    # ML-инференс и загрузка модели
│   ├── __init__.py
│   ├── optuna_params_search.py 
│   ├── train_model.py      # обучение модели
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── model_loader.py
│   └── data/
│       ├── __init__.py
│       ├── download_dataset.py
│       └── preprocessing_data.py
│       └── create_dataset.py
│
├── models/                 # обученные модели (CatBoost и др.)
│   └── best_catboost_fraud_model.cbm
│
├── notebooks/              # Jupyter notebooks для экспериментов
├── data/                   # исходные данные
├── mlartifacts/            # артефакты моделей (логирование MLflow)
├── mlruns/                 # локальное хранилище MLflow runs (если есть)
│
├── requirements.txt        # зависимости Python
├── Dockerfile             # Dockerfile для контейнеризации
├── docker-compose.yml     # опционально: docker-compose конфиг
├── .dockerignore          # файлы для игнорирования в Docker
├── .gitignore             # файлы для игнорирования в Git
├── README.md              # документация проекта
├── setup.py              # установка пакета Python
├── setup.ipynb           # вспомогательный notebook
└── tests/                # тесты
    ├── __init__.py
    ├── test_api.py
    └── test_inference.py