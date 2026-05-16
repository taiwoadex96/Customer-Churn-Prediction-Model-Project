import os
import sys
import joblib

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Saves a python object to a specified file path using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj, file_path)
        logging.info(f"Successfully saved object at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)