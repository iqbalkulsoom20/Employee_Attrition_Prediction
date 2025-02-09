# Employee_Attrition_Prediction

This project aims to predict employee attrition using machine learning. It analyzes key HR data factors and provides actionable insights for employee retention. The model is built using Logistic Regression and interpreted with LIME for explainability.

## Dataset:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?resource=download&select=WA_Fn-UseC_-HR-Employee-Attrition.csv

## Steps for process:
### 1. Data Preprocessing (EDA & Cleaning):
Loads the IBM HR Analytics dataset, checks for missing values, and performs exploratory data analysis (EDA) to understand key factors affecting employee attrition.
### 2. Training model:
Uses Logistic Regression to build a predictive model for employee attrition by splitting data into training and testing sets, evaluating model performance, and saving the trained model.
### 3. Explain Model Predictions:
Applies LIME (Local Interpretable Model-agnostic Explanations) to interpret individual predictions made by the model and helps in understanding which features contribute most to an employee’s likelihood of leaving.
### 4. Generate Actionable Insights:
Analyzes LIME explanations and model outputs to derive key retention strategies and provides HR recommendations based on data-driven insights to reduce employee attrition.
 ## Requirements:
- lime==0.2.0.1
- matplotlib==3.7.1
- numpy==1.25.0
- pandas==2.0.3
- scikit-learn==1.3.0
- seaborn==0.12.2
- joblib==1.3.2
