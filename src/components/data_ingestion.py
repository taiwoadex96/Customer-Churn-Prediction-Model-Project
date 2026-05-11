import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


class DataIngestion:

    def __init__(self):

        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw.csv')


    def initiate_data_ingestion(self):

        logger.info("Entered the data ingestion method")

        try:
            # Read dataset
            df = pd.read_csv('notebook/data/Telco-Customer-Churn.csv')

            logger.info("Dataset read successfully")

            # Create artifacts folder
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False)

            logger.info("Train-test split initiated")

            # Split data
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train and test data
            train_set.to_csv(self.train_data_path, index=False, header=True)

            test_set.to_csv(self.test_data_path, index=False, header=True)

            logger.info("Data ingestion completed")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print(train_data, test_data)