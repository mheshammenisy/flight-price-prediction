# app.py

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


# ----- paths (works locally + Streamlit Cloud) -----
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "flight_price_model.pkl"



@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def get_encoder_categories(pipeline):
    """
    Returns:
      cat_cols: list[str]
      categories_map: dict[col -> list[str]]
    pulled from the fitted OneHotEncoder inside your saved pipeline.
    """
    pre = pipeline.named_steps["preprocess"]
    cat_transformer = pre.named_transformers_["cat"]
    onehot = cat_transformer.named_steps["onehot"]

    # ColumnTransformer stores original column names in transformers_
    # We want the list of categorical columns in the same order as OneHotEncoder.categories_
    cat_cols = None
    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            cat_cols = list(cols)
            break

    categories_map = {}
    for col, cats in zip(cat_cols, onehot.categories_):
        # cats are numpy arrays; convert to list of python strings
        categories_map[col] = [str(x) for x in cats]

    return cat_cols, categories_map


pipeline = load_pipeline()

st.title("Flight Price Prediction")
st.write("Select flight details and predict the ticket price.")

# ---- get dropdown options from the trained encoder ----
cat_cols, categories_map = get_encoder_categories(pipeline)

# ---- UI: Categorical (dropdowns) ----
# These must match your engineered columns used in training
airline_options = categories_map.get("Airline", [])
source_options = categories_map.get("Source", [])
destination_options = categories_map.get("Destination", [])
addinfo_options = categories_map.get("Additional_Info", [])

airline = st.selectbox("Airline", options=airline_options, index=0 if airline_options else None)
source = st.selectbox("Source", options=source_options, index=0 if source_options else None)
destination = st.selectbox("Destination", options=destination_options, index=0 if destination_options else None)
additional_info = st.selectbox("Additional Info", options=addinfo_options, index=0 if addinfo_options else None)

# ---- UI: Numeric inputs ----
total_stops = st.number_input("Total Stops", min_value=0, max_value=4, value=1, step=1)
journey_month = st.number_input("Journey Month (1-12)", min_value=1, max_value=12, value=5, step=1)
journey_weekday = st.number_input("Journey Weekday (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=2, step=1)
dep_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=10, step=1)
arr_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=14, step=1)
duration_mins = st.number_input("Duration (minutes)", min_value=0, value=170, step=5)

# ---- Build input row (must match training feature names) ----
input_df = pd.DataFrame([{
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Total_Stops": int(total_stops),
    "Additional_Info": additional_info,
    "journey_month": int(journey_month),
    "journey_weekday": int(journey_weekday),
    "dep_hour": int(dep_hour),
    "arr_hour": int(arr_hour),
    "duration_mins": int(duration_mins),
}])

st.subheader("Input preview")
st.dataframe(input_df)

if st.button("Predict Price"):
    pred_log = float(pipeline.predict(input_df)[0])
    pred_price = float(np.expm1(pred_log))

    st.success(f"Predicted Price: {pred_price:,.0f}")
