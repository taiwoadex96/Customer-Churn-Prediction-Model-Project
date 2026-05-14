import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # =====================================================
    # MODEL EVALUATION FUNCTION
    # =====================================================

    def evaluate_model(
        self,
        y_true,
        y_pred,
        y_prob,
        model_name
    ):

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        print("\n" + "=" * 50)
        print(f"{model_name} Performance")
        print("=" * 50)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))

        return f1

    # =====================================================
    # MAIN TRAINING FUNCTION
    # =====================================================

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path
    ):

        try:

            logging.info("Starting Model Training")
            preprocessor = load_object(preprocessor_path)

            # =====================================================
            # SPLIT FEATURES AND TARGET
            # =====================================================

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            # Convert target labels to integer
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # =====================================================
            # LOGISTIC REGRESSION
            # =====================================================

            smote = SMOTE(random_state=42)

            # Apply SMOTE
            X_train_resampled_lr, y_train_resampled_lr = (
                smote.fit_resample(
                    X_train,
                    y_train
                )
            )

            logistic_model = LogisticRegression(
                max_iter=5000,
                solver='liblinear',
                random_state=42
            )

            logistic_model.fit(
                X_train_resampled_lr,
                y_train_resampled_lr
            )

            logistic_predictions = logistic_model.predict(
                X_test
            )

            logistic_probabilities = logistic_model.predict_proba(
                X_test
            )[:, 1]

            logistic_f1 = self.evaluate_model(
                y_test,
                logistic_predictions,
                logistic_probabilities,
                "Logistic Regression"
            )

            # =====================================================
            # RANDOM FOREST
            # =====================================================

            X_train_resampled_rf, y_train_resampled_rf = (
                smote.fit_resample(
                    X_train,
                    y_train
                )
            )

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }

            random_forest = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='recall',
                n_jobs=-1
            )

            random_forest.fit(
                X_train_resampled_rf,
                y_train_resampled_rf
            )

            best_rf = random_forest.best_estimator_

            random_forest_predictions = best_rf.predict(
                X_test
            )

            random_forest_probabilities = best_rf.predict_proba(
                X_test
            )[:, 1]

            rf_f1 = self.evaluate_model(
                y_test,
                random_forest_predictions,
                random_forest_probabilities,
                "Random Forest"
            )

            print("\nBest RF Parameters:")
            print(random_forest.best_params_)

            # ==========================================================
            # FEATURE IMPORTANCE
            # ==========================================================

            print("\n" + "=" * 50)
            print("TOP 10 IMPORTANT FEATURES")
            print("=" * 50)

            # Load preprocessor object
            preprocessor = load_object(preprocessor_path)

            # Get feature importances
            importances = best_rf.feature_importances_

            # Get transformed feature names
            feature_names = preprocessor.get_feature_names_out()

            # Clean feature names
            clean_feature_names = []

            for feature in feature_names:

                # Remove pipeline prefixes
                feature = feature.replace("num_pipeline__", "")
                feature = feature.replace("multi_pipeline__", "")
                feature = feature.replace("remainder__", "")

                # Make names cleaner
                feature = feature.replace("_Yes", "")
                feature = feature.replace("_No", "")
                feature = feature.replace("_Male", "")
                feature = feature.replace("_Female", "")

                clean_feature_names.append(feature)

            # Create dataframe
            feature_importance_df = pd.DataFrame({
                "Feature": clean_feature_names,
                "Importance": importances
            })

            # Sort values
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)

            # Print result
            print(feature_importance_df)

            # Plot
            plt.figure(figsize=(10, 6))

            sns.barplot(
                x="Importance",
                y="Feature",
                hue="Feature",
                data=feature_importance_df,
                palette="viridis",
                legend=False
            )

            plt.title("Top 10 Drivers of Customer Churn")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")

            plt.tight_layout()
            plt.show()

            # =====================================================
            # BEST MODEL SELECTION
            # =====================================================

            if logistic_f1 > rf_f1:

                best_model = logistic_model
                best_score = logistic_f1
                best_model_name = "Logistic Regression"

            else:

                best_model = best_rf
                best_score = rf_f1
                best_model_name = "Random Forest"

            print("\nBest Model:", best_model_name)
            print(f"Best F1 Score: {best_score:.4f}")

            # =====================================================
            # SAVE MODEL
            # =====================================================

            # Create final pipeline
            final_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", best_model)
            ])

            # Train final pipeline
            final_pipeline.fit(X_train, y_train)

            # Save full pipeline
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_pipeline
            )

            logging.info("Model Training Completed")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)


# =====================================================
# RUN PIPELINE
# =====================================================

if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Data Ingestion
    ingestion = DataIngestion()

    train_data_path, test_data_path = (
        ingestion.initiate_data_ingestion()
    )

    # Data Transformation
    transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )

    # Model Training
    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        train_arr,
        test_arr,
        preprocessor_path
    )