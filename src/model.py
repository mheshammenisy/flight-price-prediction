# src/model_factory.py

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def build_model(model_type: str = "random_forest"):
    """
    Factory function to build regression models.

    Parameters
    ----------
    model_type : str
        One of:
        - 'linear'
        - 'ridge'
        - 'random_forest'
        - 'gradient_boosting'
    """

    if model_type == "linear":
        print("-> Model: Linear Regression")
        return LinearRegression()

    elif model_type == "ridge":
        print("-> Model: Ridge Regression")
        return Ridge(alpha=1.0)

    elif model_type == "random_forest":
        print("-> Model: Random Forest Regressor (Winner)")
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        print("-> Model: Gradient Boosting Regressor")
        return GradientBoostingRegressor(
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
