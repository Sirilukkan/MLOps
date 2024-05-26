import os
import pandas as pd
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#Load dataset
def load_dataset(file_name): #need to mention file name config.TESTFILE
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

#Serialization
# def save_pipeline(pipeline_to_save, model_name):
#     save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
#     joblib.dump(pipeline_to_save, save_path)
#     print(f"Model has been save under the name {model_name}")
    
def save_pipeline(pipeline_to_save, model_key):
    try:
        #Rettrive the model name from the config using the provided key
        
        model_name = config.MODEL_NAMES[model_key]
    except KeyError:
        #Handle the case where the model_key is not in the MODEL_NAMES dictionary
        print(f"Error: The model_key '{model_key}' is not found in the MODEL_NAMES dictionary")
        return
    
    #Construct the full path to save the model
    save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {model_name}")
    
    
# #Deserialization
# def load_pipeline(pipeline_to_load):
#     save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
#     model_loaded = joblib.load(save_path)
#     print(f"Model has been loaded")
#     return model_loaded
    
#Deserialization
def load_pipeline(model_key):
    try:
        model_name = config.MODEL_NAMES[model_key]
    except KeyError:
        print(f"The model_key '{model_key}' is not found in MODEL_NAMES dictionary")
        return None
    
    save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
    
    model_loaded = joblib.load(save_path)
    print(f"Model '{model_name}' has been loaded from '{save_path}'")
    return model_loaded




