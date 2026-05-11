import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger

import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            # Numerical columns
            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            # Categorical columns
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

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logger.info("Numerical columns scaling completed")
            logger.info("Categorical columns encoding completed")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and test data loaded")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"

            # Convert TotalCharges
            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

            # Drop customerID
            train_df = train_df.drop(columns=["customerID"])
            test_df = test_df.drop(columns=["customerID"])

            # Convert target column
            train_df[target_column_name] = train_df[target_column_name].map({"No":0, "Yes":1})
            test_df[target_column_name] = test_df[target_column_name].map({"No":0, "Yes":1})

            # Features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # Save preprocessor object
            joblib.dump(
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            logger.info("Preprocessor object saved")

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

    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    print(train_arr.shape)
    print(test_arr.shape)