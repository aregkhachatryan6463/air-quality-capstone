from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.evaluation.metrics import regression_metrics

_LGBM_IMPORT_TRIED = False
_LGBM_REGRESSOR = None
_LGBM_IMPORT_ERROR: Exception | None = None
_LGBM_WARNED = False


def _get_lgbm_regressor():
    global _LGBM_IMPORT_TRIED, _LGBM_REGRESSOR, _LGBM_IMPORT_ERROR
    if _LGBM_IMPORT_TRIED:
        if _LGBM_REGRESSOR is None:
            raise RuntimeError(f"{_LGBM_IMPORT_ERROR}")
        return _LGBM_REGRESSOR
    _LGBM_IMPORT_TRIED = True
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        _LGBM_REGRESSOR = LGBMRegressor
        return _LGBM_REGRESSOR
    except Exception as ex:  # noqa: BLE001
        _LGBM_IMPORT_ERROR = ex
        _LGBM_REGRESSOR = None
        raise


def make_tree_model(model_name: str, *, random_state: int = 42, params: dict[str, Any] | None = None):
    p = dict(params or {})
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **p)
    if model_name == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=-1, **p)
    if model_name == "LightGBM":
        LGBMRegressor = _get_lgbm_regressor()
        return LGBMRegressor(random_state=random_state, **p)
    raise ValueError(model_name)


def build_baseline_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "RidgeLikePlaceholder": None,  # actual Ridge trained in pipeline for minimal deps here
        "RandomForest": make_tree_model(
            "RandomForest",
            random_state=random_state,
            params={"n_estimators": 100, "max_depth": 10},
        ),
        "XGBoost": make_tree_model(
            "XGBoost",
            random_state=random_state,
            params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        ),
    }
    try:
        models["LightGBM"] = make_tree_model(
            "LightGBM",
            random_state=random_state,
            params={"n_estimators": 200, "learning_rate": 0.05, "max_depth": -1},
        )
    except Exception as ex:  # noqa: BLE001
        global _LGBM_WARNED
        if not _LGBM_WARNED:
            warnings.warn(f"LightGBM not available, continuing without it: {ex!s}", stacklevel=2)
            _LGBM_WARNED = True
        return models
    return models


def _space_for(model_name: str) -> dict[str, Any]:
    from hyperopt import hp

    if model_name == "RandomForest":
        return {
            "n_estimators": hp.quniform("n_estimators", 100, 500, 25),
            "max_depth": hp.quniform("max_depth", 4, 20, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": hp.quniform("n_estimators", 100, 600, 25),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": hp.quniform("n_estimators", 100, 600, 25),
            "num_leaves": hp.quniform("num_leaves", 16, 128, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "min_child_samples": hp.quniform("min_child_samples", 5, 50, 1),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
        }
    raise ValueError(model_name)


def tune_hyperopt(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    max_evals: int = 40,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from hyperopt import STATUS_OK, Trials, fmin, tpe
    LGBMRegressor = None
    if model_name == "LightGBM":
        LGBMRegressor = _get_lgbm_regressor()

    space = _space_for(model_name)

    def objective(params):
        p = dict(params)
        for k in ("n_estimators", "max_depth", "min_samples_leaf", "num_leaves", "min_child_samples"):
            if k in p:
                p[k] = int(round(p[k]))
        if model_name == "RandomForest":
            model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **p)
        elif model_name == "XGBoost":
            model = XGBRegressor(random_state=random_state, n_jobs=-1, **p)
        else:
            model = LGBMRegressor(random_state=random_state, **p)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        loss = regression_metrics(y_val, pred)["MAE"]
        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_clean = {}
    for k, v in best.items():
        if k in ("n_estimators", "max_depth", "min_samples_leaf", "num_leaves", "min_child_samples"):
            best_clean[k] = int(round(v))
        else:
            best_clean[k] = float(v)
    return best_clean, {"n_trials": len(trials.trials)}

