import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configures the storage path for the preprocessor object."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df):
        """Creates and returns a ColumnTransformer based on dataframe columns."""
        try:
            logging.info("Categorizing numerical and categorical columns for pipeline assembly")
            
            # Exclude our target variable 'Churn'
            columns_to_process = [col for col in df.columns if col != 'Churn']
            
            # Identify numerical and categorical columns dynamically
            num_features = [col for col in columns_to_process if df[col].dtype != 'O']
            cat_features = [col for col in columns_to_process if df[col].dtype == 'O']

            logging.info(f"Numerical Columns: {num_features}")
            logging.info(f"Categorical Columns: {cat_features}")

            # Define Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Define Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]
            )

            # Combine Pipelines into a Single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Performs data cleaning, transformation, and applies SMOTE."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed within transformation component")

            # Custom Cleaning step matching our notebook discovery
            logging.info("Applying custom numeric conversion for TotalCharges")
            for dataset in [train_df, test_df]:
                dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
                # Drop rows where TotalCharges became null during conversion
                dataset.dropna(subset=['TotalCharges'], inplace=True)
                if 'customerID' in dataset.columns:
                    dataset.drop('customerID', axis=1, inplace=True)

            # Separate Features and Target
            input_feature_train_df = train_df.drop(columns=['Churn'], axis=1)
            target_feature_train_df = train_df['Churn'].map({'No': 0, 'Yes': 1})

            input_feature_test_df = test_df.drop(columns=['Churn'], axis=1)
            target_feature_test_df = test_df['Churn'].map({'No': 0, 'Yes': 1})

            logging.info("Obtaining preprocessing object pipeline")
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            # Fit and transform the data
            logging.info("Applying preprocessing pipeline to train and test feature matrices")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply SMOTE to training data only to balance the churn target
            logging.info("Applying SMOTE to training array to handle class imbalance")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            # Consolidate feature matrices and targets back into structural numpy arrays
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Serialize and save our preprocessing configuration pipeline
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor state saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)