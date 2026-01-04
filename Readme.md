# ✈️ Flight Price Prediction

End-to-end data science project to predict airline ticket prices using machine learning.

Overview

Performed full EDA, data cleaning, and business analysis

Prepared data for a regression task

Trained and evaluated multiple machine learning models

Deployed the final model using Streamlit Cloud

# 1. Exploratory Data Analysis (EDA)

Univariate analysis: distributions of numerical and categorical features

Bivariate / multivariate analysis: relationships between features and price

Data cleaning: fixed formats and removed irrelevant columns

Business questions: answered using numerical analysis and visualizations
(e.g. impact of airline, number of stops, and duration on price)

# 2. Data Preparation

Handled missing values and outliers

Feature engineering: date, time, and duration features

Encoded categorical variables

Applied scaling and log transformation where needed

# 3. Machine Learning

Task: Regression

Models tried: Linear Regression, Decision Tree, Random Forest

Best model: Random Forest Regressor

Evaluation metrics: RMSE, R²

# 4. Deployment

Model deployed using Streamlit Cloud

Users can input flight details and receive predicted prices

# 🔗 Live App:
https://alx-systemengineering-devops-yq2ueq2wdyatyeoazbbcuw.streamlit.app/

# Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

Run Locally
pip install -r requirements.txt
streamlit run src/app.py
