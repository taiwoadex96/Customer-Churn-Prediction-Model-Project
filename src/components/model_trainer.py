import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logger

import joblib


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred, y_prob):

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_prob)
        }

        return metrics

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logger.info("Splitting training and testing data")

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            logger.info("Applying SMOTE for class balancing")

            smote = SMOTE(random_state=42)

            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train,
                y_train
            )

            logger.info("SMOTE applied successfully")

            # Models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            best_model = None
            best_model_name = None
            best_f1_score = 0

            # Train and evaluate models
            for model_name, model in models.items():

                logger.info(f"Training {model_name}")

                model.fit(X_train_resampled, y_train_resampled)

                y_pred = model.predict(X_test)

                y_prob = model.predict_proba(X_test)[:, 1]

                metrics = self.evaluate_model(
                    y_test,
                    y_pred,
                    y_prob
                )

                print(f"\n{'='*50}")
                print(f"{model_name} Performance")
                print(f"{'='*50}")

                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")

                # Select best model using F1 Score
                if metrics["F1 Score"] > best_f1_score:

                    best_f1_score = metrics["F1 Score"]

                    best_model = model

                    best_model_name = model_name

            print(f"\nBest Model: {best_model_name}")
            print(f"Best F1 Score: {best_f1_score:.4f}")

            logger.info(f"Best model selected: {best_model_name}")

            # Save best model
            joblib.dump(
                best_model,
                self.model_trainer_config.trained_model_file_path
            )

            logger.info("Model saved successfully")

            return best_f1_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Data ingestion
    ingestion = DataIngestion()

    train_data, test_data = ingestion.initiate_data_ingestion()

    # Data transformation
    transformation = DataTransformation()

    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    # Model training
    modeltrainer = ModelTrainer()

    modeltrainer.initiate_model_trainer(
        train_arr,
        test_arr
    )