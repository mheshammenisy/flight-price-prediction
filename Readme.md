# Flight Price Prediction

End-to-end machine learning project to predict airline ticket prices.

## Project Overview
- Performed full EDA and data cleaning
- Engineered time-based and categorical features
- Trained multiple regression models
- Selected best model (Random Forest)
- Deployed model using Streamlit

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## Model
- Target: Flight Price (log-transformed)
- Best Model: Random Forest Regressor
- Metric: RMSE, R²

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
