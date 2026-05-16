import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )

    # =====================================================
    # CREATE PREPROCESSOR
    # =====================================================

    def get_data_transformer_object(self):

        try:

            logging.info(
                "Creating preprocessing pipelines"
            )

            # =====================================================
            # NUMERICAL COLUMNS
            # =====================================================

            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            # =====================================================
            # CATEGORICAL COLUMNS
            # =====================================================

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            # =====================================================
            # NUMERICAL PIPELINE
            # =====================================================

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # =====================================================
            # CATEGORICAL PIPELINE
            # =====================================================

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            # =====================================================
            # COLUMN TRANSFORMER
            # =====================================================

            preprocessor = ColumnTransformer(
                transformers=[

                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )

                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================
    # MAIN TRANSFORMATION FUNCTION
    # =====================================================

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            logging.info(
                "Reading train and test datasets"
            )

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            # =====================================================
            # CLEAN TotalCharges COLUMN
            # =====================================================

            train_df["TotalCharges"] = (
                train_df["TotalCharges"]
                .replace(" ", np.nan)
            )

            test_df["TotalCharges"] = (
                test_df["TotalCharges"]
                .replace(" ", np.nan)
            )

            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"]
            )

            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"]
            )
            # =====================================================
            # TARGET COLUMN
            # =====================================================

            target_column_name = "Churn"

            # =====================================================
            # CONVERT TARGET TO NUMERIC
            # =====================================================

            train_df[target_column_name] = (
                train_df[target_column_name]
                .map({
                    "Yes": 1,
                    "No": 0
                })
            )

            test_df[target_column_name] = (
                test_df[target_column_name]
                .map({
                    "Yes": 1,
                    "No": 0
                })
            )

            # =====================================================
            # SPLIT INPUT AND TARGET
            # =====================================================

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = (
                train_df[target_column_name]
            )

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
            )

            target_feature_test_df = (
                test_df[target_column_name]
            )

            logging.info(
                "Applying preprocessing object"
            )

            # =====================================================
            # FIT + TRANSFORM TRAIN DATA
            # =====================================================

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # =====================================================
            # TRANSFORM TEST DATA
            # =====================================================

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            # =====================================================
            # COMBINE FEATURES + TARGET
            # =====================================================

            train_arr = np.c_[

                input_feature_train_arr,
                np.array(target_feature_train_df)

            ]

            test_arr = np.c_[

                input_feature_test_arr,
                np.array(target_feature_test_df)

            ]

            logging.info(
                "Saving preprocessing object"
            )

            save_object(

                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),

                obj=preprocessing_obj

            )

            logging.info(
                "Data Transformation Completed"
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)


# =====================================================
# TEST TRANSFORMATION
# =====================================================

if __name__ == "__main__":

    from src.components.data_ingestion import (
        DataIngestion
    )

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print("Train Array Shape:", train_arr.shape)
    print("Test Array Shape:", test_arr.shape)

    print("\nPreprocessor Saved At:")
    print(preprocessor_path)