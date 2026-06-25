import json
import pandas as pd
import xgboost as xgb

# Load Best Params from the JSON bridge
# This file was created by your Optuna notebook
with open('models\\best_params.json', 'r') as f:
    best_params = json.load(f)

# Load X_train and y_train

X_train = pd.read_parquet('data\\processed\\X_train.parquet')
y_train = pd.read_parquet('data\\processed\\y_train.parquet')

# Initialize and Train
best_model = xgb.XGBClassifier(**best_params, enable_categorical=True)
best_model.fit(X_train, y_train)

# Save finished model

best_model.save_model('models\\xgboost_final.json')