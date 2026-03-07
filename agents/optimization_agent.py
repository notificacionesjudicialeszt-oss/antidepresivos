"""
OptimizationAgent — Agente de Optimización con Optuna (Fase 4).

Responsabilidades:
    1. Optimizar hiperparámetros del modelo IC₅₀ con Optuna
    2. Optimizar hiperparámetros del modelo EC₅₀ (si existe)
    3. Re-entrenar con mejores parámetros
    4. Comparar modelo default vs optimizado
"""

import logging

import numpy as np
import pandas as pd

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import model_trainer, optuna_tuner, report_generator

logger = logging.getLogger(__name__)


class OptimizationAgent(BaseAgent):
    """Agente que optimiza los hiperparámetros de los modelos con Optuna."""

    def __init__(self, cfg=None):
        super().__init__(name="OptimizationAgent", config=cfg or config)

    def validate_inputs(self) -> None:
        if not self.config.FEATURES_MATRIX.exists():
            raise ValidationError("Matriz de features no encontrada.")
        if not self.config.MODELO_IC50.exists():
            raise ValidationError(
                "Modelo IC₅₀ base no encontrado. Ejecuta primero el TrainingAgent."
            )

    def execute(self) -> dict:
        """Optimiza hiperparámetros y re-entrena."""

        # Cargar datos
        df = pd.read_csv(self.config.FEATURES_MATRIX)
        feature_cols = [c for c in df.columns if c not in ("pActivity", "canonical_smiles")]
        X = df[feature_cols].values
        y = df["pActivity"].values

        # ── Optimizar IC₅₀ ──
        self.logger.info("═══ Optimizando IC₅₀ con Optuna ═══")
        opt_result = optuna_tuner.optimize_hyperparameters(
            X, y,
            n_trials=self.config.OPTUNA_TRIALS,
            cv_folds=self.config.CV_FOLDS,
            random_state=self.config.RANDOM_STATE,
            search_space=self.config.OPTUNA_SEARCH_SPACE,
        )

        # Re-entrenar con mejores params
        self.logger.info("═══ Re-entrenando IC₅₀ con parámetros óptimos ═══")
        optimized = model_trainer.train_model(
            X, y,
            params=opt_result["best_params"],
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )
        model_trainer.save_model(optimized["model"], self.config.MODELO_IC50_OPTUNA)

        # Cargar métricas del modelo base para comparar
        base_result = model_trainer.train_model(
            X, y,
            params=self.config.XGBOOST_DEFAULTS,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )

        return {
            "best_params": opt_result["best_params"],
            "best_rmse": opt_result["best_rmse"],
            "n_trials": opt_result["n_trials"],
            "optimized_metrics": optimized["metrics"],
            "base_metrics": base_result["metrics"],
        }

    def validate_outputs(self, data: dict) -> None:
        if not self.config.MODELO_IC50_OPTUNA.exists():
            raise ValidationError("Modelo optimizado no fue guardado")

        opt_r2 = float(data["optimized_metrics"]["R²"])
        base_r2 = float(data["base_metrics"]["R²"])

        if opt_r2 < base_r2:
            self.logger.warning(
                f"⚠️  Modelo optimizado (R²={opt_r2:.4f}) no superó al base (R²={base_r2:.4f}). "
                f"Esto puede ocurrir por overfitting en la optimización."
            )

    def generate_report(self, data: dict) -> None:
        report_generator.generate_report(
            title="⚡ Reporte de Optimización — Fase 4",
            agent_name=self.name,
            sections={
                "Configuración": {
                    "Trials Optuna": data["n_trials"],
                    "Mejor RMSE (CV)": f"{data['best_rmse']:.4f}",
                },
                "Mejores Hiperparámetros": {
                    k: f"{v}" for k, v in data["best_params"].items()
                },
                "Comparación: Default vs Optimizado": {
                    "Default R²": data["base_metrics"]["R²"],
                    "Optimizado R²": data["optimized_metrics"]["R²"],
                    "Default RMSE": data["base_metrics"]["RMSE"],
                    "Optimizado RMSE": data["optimized_metrics"]["RMSE"],
                },
                "Modelo Guardado": [
                    f"`{self.config.MODELO_IC50_OPTUNA}`",
                ],
            },
            output_path=self.config.REPORTES_DIR / "reporte_optimizacion.md",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = OptimizationAgent()
    result = agent.run()
    print(result)
