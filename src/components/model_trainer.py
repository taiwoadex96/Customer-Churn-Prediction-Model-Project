import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configures the storage path for the optimized model artifact."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Trains, tunes, and serializes the highest performing model."""
        try:
            logging.info("Splitting training and test input arrays into feature matrices and targets")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models to experiment with
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            # Define hyperparameter grid configurations matching our notebook
            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1.0, 10.0]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                }
            }

            # Run evaluation loop via utils
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Find the best model name and score from our report dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Enforce a quality threshold before saving models to production
            if best_model_score < 0.60:
                raise CustomException("No model met the baseline quality threshold of 60% Recall.", sys)
            
            logging.info(f"Winning Model Identified: {best_model_name} with a Recall score of {best_model_score:.4f}")
            print(f"\n Winning Model Selected: {best_model_name} (Recall: {best_model_score:.4f})")

            # Save the winning estimator model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Print out out a complete evaluation log for validation
            predictions = best_model.predict(X_test)
            print("\nFinal Model Evaluation Performance Matrix:")
            print(classification_report(y_test, predictions))

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)