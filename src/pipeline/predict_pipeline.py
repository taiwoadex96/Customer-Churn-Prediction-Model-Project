import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            # =====================================================
            # LOAD MODEL
            # =====================================================

            model_path = "artifacts/model.pkl"

            model = load_object(model_path)

            # =====================================================
            # LOAD PREPROCESSOR
            # =====================================================

            preprocessor_path = "artifacts/preprocessor.pkl"

            preprocessor = load_object(preprocessor_path)

            # =====================================================
            # TRANSFORM INPUT DATA
            # =====================================================

            data_transformed = preprocessor.transform(
                features
            )

            # =====================================================
            # PREDICTION
            # =====================================================

            predictions = model.predict(
                data_transformed
            )

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# =====================================================
# CUSTOM INPUT CLASS
# =====================================================

class CustomData:

    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float
    ):

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

    # =====================================================
    # CONVERT INPUT TO DATAFRAME
    # =====================================================

    def get_data_as_dataframe(self):

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

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)