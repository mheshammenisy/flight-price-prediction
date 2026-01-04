# src/train.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.datapipeline import build_featured_dataset_from_excel
from src.model import build_model


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_preprocess(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_train.select_dtypes(include="object").columns
    num_cols = X_train.select_dtypes(exclude="object").columns

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds_log = pipe.predict(X_test)

    out = {
        "MAE_log": mean_absolute_error(y_test, preds_log),
        "RMSE_log": rmse(y_test, preds_log),
        "R2_log": r2_score(y_test, preds_log),
    }

    # optional: real price metrics for report/video
    y_price = np.expm1(y_test)
    pred_price = np.expm1(preds_log)

    out.update({
        "MAE_price": mean_absolute_error(y_price, pred_price),
        "RMSE_price": rmse(y_price, pred_price),
        "R2_price": r2_score(y_price, pred_price),
    })

    return out


def main():
    raw_path = r"C:\flight_price_prediction\data\raw\Data_Train.xlsx"
    models_dir = r"C:\flight_price_prediction\models"
    os.makedirs(models_dir, exist_ok=True)

    # 1) data pipeline
    df = build_featured_dataset_from_excel(raw_path)

    # 2) X/y
    y = df["Price_log"]
    X = df.drop(columns=["Price", "Price_log"], errors="ignore")

    # 3) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) preprocess
    preprocess = build_preprocess(X_train)

    # 5) model types (from src/model.py)
    model_types = ["linear", "ridge", "random_forest", "gradient_boosting"]

    results = []
    trained = {}

    for mtype in model_types:
        reg = build_model(mtype)

        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("regressor", reg),
        ])

        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)

        row = {"Model": mtype, **metrics}
        results.append(row)
        trained[mtype] = pipe

        print(f"{mtype} -> RMSE_log={metrics['RMSE_log']:.4f}, R2_log={metrics['R2_log']:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE_log").reset_index(drop=True)

    print("\n=== Model Comparison (sorted by RMSE_log) ===")
    print(results_df)

    # 6) best model
    best_type = results_df.loc[0, "Model"]
    best_pipe = trained[best_type]

    # 7) save
    model_path = os.path.join(models_dir, "flight_price_model.pkl")
    metrics_path = os.path.join(models_dir, "metrics.csv")

    joblib.dump(best_pipe, model_path)
    results_df.to_csv(metrics_path, index=False)

    print(f"\nBEST MODEL: {best_type}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
