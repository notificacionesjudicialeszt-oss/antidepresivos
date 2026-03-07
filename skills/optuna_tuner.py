"""
Skill: Optuna Tuner
Optimización bayesiana de hiperparámetros de XGBoost con Optuna.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("⚠️  Optuna no está instalado. Ejecuta: pip install optuna")


def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    search_space: Optional[dict] = None,
) -> dict:
    """
    Optimización bayesiana de hiperparámetros con Optuna.

    Args:
        X: Matriz de features.
        y: Vector target.
        n_trials: Número de trials (paper: 50).
        cv_folds: Folds para cross-validation.
        random_state: Semilla.
        search_space: Rangos de búsqueda (usa defaults del paper si None).

    Returns:
        Dict con: best_params, best_score, study (objeto Optuna).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está instalado. Ejecuta: pip install optuna")

    # Defaults del paper
    if search_space is None:
        search_space = {
            "n_estimators": (100, 1000),
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "min_child_weight": (1, 10),
            "gamma": (0.0, 5.0),
            "reg_alpha": (0.0, 10.0),
            "reg_lambda": (0.0, 10.0),
        }

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", *search_space["n_estimators"]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", *search_space["max_depth"]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", *search_space["subsample"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *search_space["min_child_weight"]
            ),
            "gamma": trial.suggest_float(
                "gamma", *search_space["gamma"]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *search_space["reg_alpha"]
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search_space["reg_lambda"]
            ),
            "random_state": random_state,
            "n_jobs": -1,
        }

        model = XGBRegressor(**params)
        scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return -scores.mean()  # Minimizar RMSE

    logger.info(f"🔎 Iniciando optimización: {n_trials} trials, {cv_folds}-fold CV...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["random_state"] = random_state
    best["n_jobs"] = -1

    logger.info(f"🏆 Mejor RMSE: {study.best_value:.4f}")
    logger.info(f"🏆 Mejores parámetros: {best}")

    return {
        "best_params": best,
        "best_rmse": study.best_value,
        "study": study,
        "n_trials": n_trials,
    }
