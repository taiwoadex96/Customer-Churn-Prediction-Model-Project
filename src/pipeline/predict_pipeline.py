import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import joblib

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """Loads serialized pipeline assets to predict on unseen web forms."""
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Loading production Preprocessor and Model weights")
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            
            logging.info("Transforming incoming feature data matrices")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Executing forward classification pass")
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """Maps HTML web input elements into a structured Pandas DataFrame."""
    def __init__(self, gender: str, SeniorCitizen: int, Partner: str, Dependents: str,
                 tenure: int, PhoneService: str, MultipleLines: str, InternetService: str,
                 OnlineSecurity: str, OnlineBackup: str, DeviceProtection: str, TechSupport: str,
                 StreamingTV: str, StreamingMovies: str, Contract: str, PaperlessBilling: str,
                 PaymentMethod: str, MonthlyCharges: float, TotalCharges: float):
        
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Web application form mapped to DataFrame successfully")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)