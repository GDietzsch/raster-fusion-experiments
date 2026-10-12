"""Estimator factory; optional models fail independently when unavailable."""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor


def make_model(name: str, seed: int, params: dict | None = None, device: str = "auto"):
    params = dict(params or {})
    if name == "linear":
        return LinearRegression(**params)
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    if name == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, **params)
    if name == "hist_gradient_boosting":
        params.setdefault("early_stopping", False)
        return HistGradientBoostingRegressor(random_state=seed, **params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        from src.utils.hardware import select_device
        selected = str(select_device(device))
        params.setdefault("tree_method", "hist")
        params.setdefault("device", selected)
        return XGBRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown tabular model: {name}")
