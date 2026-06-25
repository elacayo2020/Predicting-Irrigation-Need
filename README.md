# Irrigation Prediction (XGBoost + Optuna)

## Overview
This project predicts irrigation needs using environmental and soil-related features (target y). The workflow focuses on data cleaning, exploratory data analysis, visualization, and building an optimized machine learning model. The data set was from a Kaggle competition: https://www.kaggle.com/competitions/playground-series-s6e4/overview

## Data Cleaning
Data preprocessing steps included ensuring correct data types & preparing features for modeling. All objects were turned into categories for the XGBoost model.

## Exploratory Data Analysis & Visualizations
Visualizations were used to better understand the data and relationships between features. These included box plots for numberical variable vs. the target y labels, distribution plots for numerical variables, correlation map, and bar plots for categorical variables vs. the target y lables.

### Boxplots
<img width="1189" height="1190" alt="image" src="https://github.com/user-attachments/assets/1a27e898-327d-4578-b022-69a2c6e4fb96" />
With the boxplots we noticed irrigation needs are higher for samples with dry soil, high temperatures, low rainfaill, or high wind speed.

### Barplots
<img width="1187" height="925" alt="image" src="https://github.com/user-attachments/assets/66a914a5-1c43-4f0e-aa19-ed7fdc80e432" />
We notice that most of the categorical features are consistent with the ratios of each label in irrigation needs. Samples with no mulching tend to have higher irrigation needs. Samples in the Vegetative and Flowering stages  also tend to have higher irrigation needs as well.

## Model: XGBoost
An XGBoost classifier was used as the primary model. Optuna was used to tune key XGBoost hyperparameters, improving model performance by optimizing parameters such as learning rate, max depth, and number of estimators.

### Best Hyperparameters (Optuna)

| Parameter            | Value      |
|---------------------|-----------|
| max_depth           | 7         |
| learning_rate       | 0.0597    |
| n_estimators        | 670       |
| subsample           | 0.7761    |
| colsample_bytree    | 0.5922    |
| min_child_weight    | 5         |
| gamma               | 0.2546    |

## Final Performance

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|-------------|----------|--------|----------|---------|
| 0           | 0.99     | 0.99   | 0.99     | 73,737  |
| 1           | 0.98     | 0.98   | 0.98     | 48,014  |
| 2           | 0.97     | 0.91   | 0.94     | 4,249   |
| **Accuracy**|          |        | **0.98** | 126,000 |
| **Macro Avg** | 0.98   | 0.96   | 0.97     | 126,000 |
| **Weighted Avg** | 0.98 | 0.98  | 0.98     | 126,000 |


** 98.46269841269841% on testing set

**96.025% Accuracy on Kaggle test set**