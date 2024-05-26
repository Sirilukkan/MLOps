## We treat all prediction_model as a package

import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent #prediction_model.__file__ will give path to __init__.py

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")


TRAIN_FILE = 'onlinefoods.csv'

#In this case we use the same, becuase original data did not have the test file, we have this as a place holder for test file
#But we can also split from original data, but will be without the feedback
TEST_FILE = 'onlinefoods_test.csv' 

#MODEL_NAME = 'feedback_classification.pkl' We have 4 models, so I need to think how can I create model name of 4 models

MODEL_NAMES = {
    'logistic': 'logistic_regression_model.pkl',
    'decision_tree': 'decision_tree_model.pkl',
    'random_forest': 'random_forest_model.pkl',
    'xgboost': 'xgboost_model.pkl'
}

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'Feedback'

#Final features used in the model
FEATURES = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
       'Educational Qualifications', 'Family size', 'latitude', 'longitude',
       'Pin code', 'Output']

NUM_FEATURES = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']

CAT_FEATURES = ['Gender',
 'Marital Status',
 'Occupation',
 'Monthly Income',
 'Educational Qualifications',
 'Output'] 

# in our case it is same as Categorical features + Target column that we also want to encode
FEATURES_TO_ENCODE = ['Gender',
 'Marital Status',
 'Occupation',
 'Monthly Income',
 'Educational Qualifications',
 'Output'] #We also need to encode the Feedback

LOG_FEATURES = ['Pin code'] # taking log of numerical columns
