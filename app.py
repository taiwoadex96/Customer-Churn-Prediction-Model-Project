from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Target Route for standard rendering
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_dataframe():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Harvest incoming form variables
        data = CustomData(
            gender=request.form.get('gender'),
            SeniorCitizen=int(request.form.get('SeniorCitizen')),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=int(request.form.get('tenure')),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges'))
        )
        
        # Format dictionary collection to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("\nIncoming Inference DataFrame Form:")
        print(pred_df)

        # Call the structural execution client
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Translate binary values back into human operational alerts
        final_verdict = (
            "⚠️ High Risk! This client is highly likely to CHURN." 
            if results[0] == 1 else 
            "✅ Safe! This customer is projected to STAY."
        )
        
        return render_template('home.html', results=final_verdict)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)