import os
import sys
import joblib
from sklearn.metrics import recall_score

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
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Performs grid search tuning across multiple models and scores them on Recall."""
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Starting Hyperparameter Tuning for: {model_name}")
            gs = GridSearchCV(model, para, cv=3, scoring='recall', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Assign optimized parameters back to the estimator
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on unseen testing features
            y_test_pred = model.predict(X_test)

            # Score using Recall because catching churners is our core objective
            test_model_score = recall_score(y_test, y_test_pred)
            logging.info(f"{model_name} Best Params: {gs.best_params_} | Test Recall: {test_model_score:.4f}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)