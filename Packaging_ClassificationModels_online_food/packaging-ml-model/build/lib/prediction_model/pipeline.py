from sklearn.pipeline import Pipeline

from pathlib import Path
import sys
import os

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

# Define the shared preprocessing steps
preprocessor = Pipeline([
    ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
    ('MinMaxScale', MinMaxScaler())
])

# Define individual pipelines with shared preprocessing steps
pipelines = {
    'logistic': Pipeline([
        ('Preprocessor', preprocessor),
        ('Classifier', LogisticRegression(random_state=0))
    ]),
    'decision_tree': Pipeline([
        ('Preprocessor', preprocessor),
        ('Classifier', DecisionTreeClassifier(random_state=0))
    ]),
    'random_forest': Pipeline([
        ('Preprocessor', preprocessor),
        ('Classifier', RandomForestClassifier(random_state=0))
    ]),
    'xgboost': Pipeline([
        ('Preprocessor', preprocessor),
        ('Classifier', xgb.XGBClassifier(random_state=0))
    ])
}

# Define parameter grids for each classifier
param_grid_logistic = {
    'Classifier__C': [100, 10, 1.0, 0.1, 0.01],
    'Classifier__penalty': ['l1', 'l2'],
    'Classifier__solver': ['liblinear']
}

param_grid_decision_tree = {
    "Classifier__max_depth": [3, 5, 7, 9, 11, 13],
    'Classifier__criterion': ["gini", "entropy"],
}

param_grid_random_forest = {
    'Classifier__n_estimators': [200, 400, 700],
    'Classifier__max_depth': [10, 20, 30],
    'Classifier__criterion': ["gini", "entropy"],
    'Classifier__max_leaf_nodes': [50, 100]
}

param_grid_xgboost = {
    'Classifier__max_depth': [3, 5, 7, 9, 11, 13],
    'Classifier__learning_rate': [0.1, 0.01, 0.001],
    'Classifier__n_estimators': [100, 200, 300],
    'Classifier__objective': ['binary:logistic'],
    'Classifier__eval_metric': ['error']
}


#Note: define each pipeline for each model separately ensure;
# 1. flexibility, 2. modularity, 3.interchangeability, and Performance