# Churn Prediction Intelligence System

## Problem Statement
Customer churn refers to customers who cancel or stop using a company's service. Losing customers is significantly more expensive than acquiring new ones — studies show retention is 5-7x cheaper than acquisition. This system predicts which customers are likely to churn before they leave, enabling businesses to intervene proactively with targeted retention offers and discounts.

## Dataset
This project uses the **Telco Customer Churn** dataset from IBM/Kaggle.
- **Size:** 7,043 customers, 21 features
- **Features:** Demographics, account information, and services subscribed
- **Target variable:** `Churn` (Yes/No)
- [Download from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Tech Stack
| Tool | Purpose |
|------|---------|
| **pandas** | Data manipulation, cleaning, and analysis |
| **numpy** | Numerical computing and array operations |
| **matplotlib / seaborn** | Exploratory Data Analysis (EDA) and data visualization |
| **scikit-learn** | Machine learning algorithms, data preprocessing, and evaluation metrics |
| **XGBoost** | Advanced gradient boosting machine learning models |
| **SHAP** | Model interpretability and feature importance analysis |
| **FastAPI** | Building the REST API to serve model predictions |
| **joblib** | Serializing and saving the trained machine learning models |

## Model Performance
In churn prediction, identifying the actual churners is critical, so we prioritized **Recall for the Churn class (1)** while keeping an eye on the overall F1-Score and ROC-AUC. 

| Model | Recall (Churn) | F1 (Churn) | ROC-AUC |
|-------|---------------|------------|---------|
| Logistic Regression | 0.56 | 0.61 | 0.8419 |
| Logistic Regression (Balanced) | **0.78** | **0.61** | **0.8415** |
| Random Forest | 0.50 | 0.56 | 0.8290 |
| Random Forest (Balanced) | 0.51 | 0.57 | 0.8274 |
| XGBoost | 0.50 | 0.54 | 0.8224 |
| XGBoost (Balanced) | 0.68 | 0.61 | 0.8210 |

**Selected Model:** Logistic Regression (Balanced) — It yielded the highest Recall (0.78) and a strong ROC-AUC (0.8415), meaning it is the most effective at catching customers who are actually at risk of leaving.

## Key Findings
- **Contract Type is Crucial:** Customers on month-to-month contracts exhibit the highest churn rates compared to those on one-year or two-year contracts.
- **Early Attrition:** The highest risk period is early in the customer lifecycle; users with a tenure of **0 to 10 months** are the most likely to churn.
- **Service Dependency:** Customers subscribing to Fiber Optic internet services show a significantly higher churn rate compared to DSL or no internet users.
- **Feature Importance:** SHAP (SHapley Additive exPlanations) analysis confirmed that `tenure` is the top predictive feature driving the model's decisions, followed closely by contract type and internet service.

## Sample API Request
```json
POST /predict
{
  "gender": 0,
  "SeniorCitizen": 1,
  "Partner": 0,
  "Dependents": 0,
  "tenure": 2,
  "PhoneService": 1,
  "MultipleLines": 0,
  "OnlineSecurity": 0,
  "OnlineBackup": 0,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "PaperlessBilling": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaymentMethod_Credit_card_automatic": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0
}
**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8253,
  "risk_level": "High"
}

## How to Run
1. Clone the repository
````bash
   git clone https://github.com/Vashisht-Kshitij/churn-prediction.git
   cd churn-prediction
````
2. Install dependencies
````bash
   pip install -r requirements.txt
````
3. Run the API
````bash
   uvicorn src/api/main:app --reload
````
4. Open `http://127.0.0.1:8000/docs` to test the API interactively

## Project Structure

churn-prediction/
├── data/
│   ├── raw/          # Original dataset
│   └── processed/    # Cleaned and encoded data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap.ipynb
├── src/api/
│   └── main.py       # FastAPI prediction endpoint
├── models/           # Saved model artifacts
├── reports/          # EDA and SHAP visualizations
└── requirements.txt
