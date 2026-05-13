import sys
import os

import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for:
        - Cleaning
        - Encoding
        - Creating preprocessing object

        NOTE:
        We DO NOT apply scaling here.
        Scaling will be done separately inside model_trainer.py
        only for Logistic Regression.
        """

        try:
            logging.info("Creating preprocessing object")

            # Binary categorical columns
            binary_cols = [
                'gender',
                'Partner',
                'Dependents',
                'PhoneService',
                'PaperlessBilling',
                'Churn'
            ]

            # Multi-category columns
            multi_cols = [
                'MultipleLines',
                'InternetService',
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies',
                'Contract',
                'PaymentMethod'
            ]

            # Numerical columns
            numerical_cols = [
                'SeniorCitizen',
                'tenure',
                'MonthlyCharges',
                'TotalCharges'
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )

            # Multi-category pipeline
            multi_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first'))
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("multi_pipeline", multi_pipeline, multi_cols)
                ],
                remainder='passthrough'
            )

            logging.info("Preprocessing object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            # ==========================================
            # DATA CLEANING
            # ==========================================

            # Convert TotalCharges to numeric
            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )

            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )

            # Drop missing values
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            # Drop customerID
            train_df = train_df.drop(columns=['customerID'])
            test_df = test_df.drop(columns=['customerID'])

            logging.info("Data cleaning completed")

            target_column_name = "Churn"

            # ==========================================
            # LABEL ENCODING FOR BINARY COLUMNS
            # ==========================================

            binary_cols = [
                col for col in train_df.columns
                if train_df[col].nunique() == 2
            ]

            binary_mapping = {
                'Yes': 1,
                'No': 0,
                'Male': 1,
                'Female': 0
            }

            for col in binary_cols:
                train_df[col] = train_df[col].replace(binary_mapping)
                test_df[col] = test_df[col].replace(binary_mapping)

            logging.info("Binary encoding completed")

            # ==========================================
            # SPLIT FEATURES AND TARGET
            # ==========================================

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[target_column_name]

            logging.info("Feature-target split completed")

            preprocessing_obj = self.get_data_transformer_object()

            # ==========================================
            # APPLY TRANSFORMATIONS
            # ==========================================

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            logging.info("Preprocessing completed")

            # ==========================================
            # COMBINE FEATURES + TARGET
            # ==========================================

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Train and test arrays created")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()

    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    print("Data transformation completed successfully")