from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and feature names
model = joblib.load('../../models/churn_model.pkl')
scaler = joblib.load('../../models/scaler.pkl')
feature_names = joblib.load('../../models/feature_names.pkl')

# Define input schema with valid Python names
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    # Convert to list in correct order
    values = [
        customer.gender,
        customer.SeniorCitizen,
        customer.Partner,
        customer.Dependents,
        customer.tenure,
        customer.PhoneService,
        customer.MultipleLines,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies,
        customer.PaperlessBilling,
        customer.MonthlyCharges,
        customer.TotalCharges,
        customer.InternetService_Fiber_optic,
        customer.InternetService_No,
        customer.Contract_One_year,
        customer.Contract_Two_year,
        customer.PaymentMethod_Credit_card_automatic,
        customer.PaymentMethod_Electronic_check,
        customer.PaymentMethod_Mailed_check
    ]
    
    # Build DataFrame with exact training column names
    data = pd.DataFrame([values], columns=feature_names)
    
    # Scale and predict
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }
