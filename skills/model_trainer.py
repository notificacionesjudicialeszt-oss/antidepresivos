"""
Skill: Model Trainer
Entrenamiento XGBoost con métricas completas y serialización.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """
    Entrena un XGBRegressor y evalúa con métricas completas.

    Args:
        X: Matriz de features.
        y: Vector de target (pActivity).
        params: Hiperparámetros para XGBoost (si None, usa defaults).
        test_size: Proporción de test set.
        random_state: Semilla.
        cv_folds: Número de folds para cross-validation.

    Returns:
        Diccionario con: model, metrics, predictions, splits.
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"📐 Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # Defaults si no se pasan params
    if params is None:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
        }

    # Entrenar
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Predecir en test
    y_pred = model.predict(X_test)

    # Métricas en test
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"📊 Test metrics — R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(
        XGBRegressor(**params), X, y,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1,
    )

    logger.info(
        f"📊 CV ({cv_folds}-fold) — R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    return {
        "model": model,
        "metrics": {
            "R²": f"{r2:.4f}",
            "RMSE": f"{rmse:.4f}",
            "MAE": f"{mae:.4f}",
            "CV_R²_mean": f"{cv_scores.mean():.4f}",
            "CV_R²_std": f"{cv_scores.std():.4f}",
        },
        "predictions": {
            "y_test": y_test,
            "y_pred": y_pred,
        },
        "splits": {
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
        },
    }


def save_model(model, path: Path) -> None:
    """Serializa modelo con joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"💾 Modelo guardado: {path}")


def load_model(path: Path):
    """Carga modelo serializado."""
    model = joblib.load(path)
    logger.info(f"📂 Modelo cargado: {path}")
    return model
