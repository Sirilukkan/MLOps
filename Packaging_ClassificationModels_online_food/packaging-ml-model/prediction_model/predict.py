import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, load_pipeline


#Load all pipeline
pipelines = {}
for model_key in config.MODEL_NAMES.keys():
    pipelines[model_key] = load_pipeline(model_key)

#Preliminary Testing
def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    
    predictions = {}
    for model_key, pipeline in pipelines.items():
        if pipeline is not None:
            pred = pipeline.predict(test_data[config.FEATURES])
            output = np.where(pred == 1, 'Positive', 'Negative')
            predictions[model_key] = output
        else:
            print(f"Pipeline for model '{model_key}' is not loaded.")
    
    return predictions

## Test in pytest

# # Specify the model key to use
# MODEL_KEY_TO_TEST = 'logistic'  # Replace with the actual model key you want to test

# # Load the specified pipeline
# pipeline = load_pipeline(MODEL_KEY_TO_TEST)

# def generate_predictions(data_input):
#     data = pd.DataFrame(data_input)
#     pred = pipeline.predict(data[config.FEATURES])
#     output = np.where(pred==1,'Positive','Negative')
#     result = {"predictions":output}
#     return result

if __name__ == '__main__':
    predictions = generate_predictions()
    for model_key, prediction in predictions.items():
        print(f"Predictions for {model_key}: {prediction}")