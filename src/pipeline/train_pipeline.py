from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def start_training_pipeline(self):

        # ==============================
        # DATA INGESTION
        # ==============================

        ingestion = DataIngestion()

        train_data_path, test_data_path = (
            ingestion.initiate_data_ingestion()
        )

        print("Data Ingestion Completed")

        # ==============================
        # DATA TRANSFORMATION
        # ==============================

        transformation = DataTransformation()

        train_arr, test_arr, _ = (
            transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
        )

        print("Data Transformation Completed")

        # ==============================
        # MODEL TRAINING
        # ==============================

        trainer = ModelTrainer()

        trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        print("Model Training Completed")


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.start_training_pipeline()