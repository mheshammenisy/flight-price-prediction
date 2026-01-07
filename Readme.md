# ✈️ Flight Price Prediction – Machine Learning Project

Streamlit App | Python 3.10 | Scikit-learn | Random Forest

An end-to-end machine learning project that predicts domestic flight prices in India using historical flight data.
The project covers EDA, data cleaning, feature engineering, model training, evaluation, and deployment on Streamlit Cloud.

# 📊 Model Performance
Model	R² (Test)	MAE (₹)	RMSE (₹)
Random Forest	0.88	~884	~1826
Gradient Boosting	0.83	~1399	~2327
Ridge Regression	0.75	~1732	~2756
Linear Regression	0.75	~1732	~2758

# Best Model: Random Forest Regressor

# 🔍 Key Insights (from EDA)

Airline type has a strong influence on ticket price

Flights with more stops tend to be more expensive

Early-morning and late-night flights are generally cheaper

Travel month and weekday affect pricing patterns

# 🧠 Feature Engineering

Duration: "2h 50m" → 170 minutes

Stops: "non-stop" → 0, "1 stop" → 1

Date: extracted day, month, weekday

Time: extracted departure hour and arrival hour

# ⚙️ Machine Learning Pipeline

The project uses a single Scikit-learn Pipeline:

Pipeline(
  preprocessing (ColumnTransformer)
  → model (RandomForestRegressor)
)


This ensures:

No data leakage

Safe serialization

Easy deployment

# 🧪 Models Evaluated

Linear Regression

Ridge Regression

Gradient Boosting Regressor

Random Forest Regressor ✅ (selected)

# 🚀 Live Demo

🔗 Streamlit App: 
https://alx-systemengineering-devops-yq2ueq2wdyatyeoazbbcuw.streamlit.app/

🗂️ Project Structure
flight_price_prediction/
├── src/
│   ├── datapipeline.py
│   ├── model.py
│   ├── train.py
│   ├── app.py
│   └── __init__.py
├── models/
│   └── flight_price_model.pkl
├── 01_eda.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

# 🛠️ How to Run Locally
git clone https://github.com/mheshammenisy/flight-price-prediction.git
cd flight-price-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py

# 👤 Author

Mohamed Hesham Sayed
Master’s Student – Energy & Data Analysis

📝 License

This project is licensed under the MIT License.
