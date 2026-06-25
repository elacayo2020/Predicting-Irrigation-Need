import json
import optuna
import optuna.visualization as vis
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# load the training data
X_train = pd.read_parquet('data\\processed\\X_train.parquet')
y_train = pd.read_parquet('data\\processed\\y_train.parquet')

# optimize hardware usage for Optuna
# For users WITHOUT a GPU: Set DEVICE='cpu' and N_JOBS=-1
# For users WITH a GPU: Set DEVICE='cuda' and N_JOBS=1

DEVICE = 'cpu'  # Options: 'cpu', 'cuda'
N_JOBS = -1      # Use -1 for CPU to use all cores, 1 for GPU
N_TRIALS = 2   # Total iterations for finding best hyperparameters

# Defining the objective function
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    # initializing the XGBoost model
    model = xgb.XGBClassifier(**param,
                              tree_method = 'hist',
                              device =  DEVICE,
                              enable_categorical = True)

    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score

# Create and run the optimization process
study = optuna.create_study(study_name="example_xgboost_study", direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

# Retrieve the best parameter values
best_params = study.best_params

# Save as a JSON file
with open('models\\best_params.json', 'w') as f:
    json.dump(best_params, f)

print(f"\nBest parameters: {best_params}")