import pandas as pd
import numpy as np 
from pathlib import Path
import sys
import os

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config 
from prediction_model.processing.data_handling import load_dataset, save_pipeline
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe 
from prediction_model.pipeline import param_grid_logistic, param_grid_decision_tree, param_grid_random_forest, param_grid_xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



def perform_training():
    # Load dataset
    train_data = load_dataset(config.TRAIN_FILE)
    
    # Extract features and target variable
    X = train_data[config.FEATURES]
    y = train_data[config.TARGET].map({'Positive': 1, 'Negative ': 0})  # Note: fix space issue in 'Negative'
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define parameter grids for each classifier
    param_grids = {
        'logistic': param_grid_logistic,
        'decision_tree': param_grid_decision_tree,
        'random_forest': param_grid_random_forest,
        'xgboost': param_grid_xgboost
    }
        
    # Perform grid search for each classifier
    for model_key, pipeline in pipe.pipelines.items():
        param_grid = param_grids[model_key]
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Save the best model from grid search
        best_model = grid_search.best_estimator_
        save_pipeline(best_model, model_key)

if __name__ == '__main__':
    perform_training()
