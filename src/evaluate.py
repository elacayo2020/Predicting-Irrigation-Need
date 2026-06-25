import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load test data
X_test = pd.read_parquet('data\\processed\\X_test.parquet')
y_test = pd.read_parquet('data\\processed\\y_test.parquet')

# Load the trained model
best_model = xgb.XGBClassifier()
best_model.load_model('models\\xgboost_final.json')

# Predict the test data
y_pred = best_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate a classification report
cr = classification_report(y_test, y_pred)
print(f"\nReport:\n{cr}")