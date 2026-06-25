import pandas as pd
import pyarrow
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv('data\\raw\\train.csv', index_col=0)
kaggle_test = pd.read_csv('data\\raw\\test.csv', index_col=0)

# Convert all object columns to category
for col in train_df.select_dtypes('object').columns:
    train_df[col] = train_df[col].astype('category')
    if col in kaggle_test.columns:
        kaggle_test[col] = kaggle_test[col].astype('category')

# Encode irrigation needs
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
train_df['Irrigation_Need'] = train_df['Irrigation_Need'].map(mapping)

# Split data for testing
X = train_df.drop(columns='Irrigation_Need')
y = train_df['Irrigation_Need']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
X_train.to_parquet('data\\processed\\X_train.parquet', index=False)
X_test.to_parquet('data\\processed\\X_test.parquet', index=False)
y_train.to_frame().to_parquet('data\\processed\\y_train.parquet', index=False)
y_test.to_frame().to_parquet('data\\processed\\y_test.parquet', index=False)
kaggle_test.to_parquet('data\\processed\\cleaned_kaggle_test.parquet', index=False)

print(X_train.info())