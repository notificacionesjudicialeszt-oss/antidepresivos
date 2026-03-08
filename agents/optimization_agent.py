"""
OptimizationAgent — Agente de Optimización con Optuna (Fase 4).

Soporta múltiples targets (SERT, NAT, DAT).
"""

import logging

import numpy as np
import pandas as pd

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import model_trainer, optuna_tuner, report_generator

logger = logging.getLogger(__name__)


class OptimizationAgent(BaseAgent):
    """Agente que optimiza hiperparámetros con Optuna para un target."""

    def __init__(self, target_key: str = None, cfg=None):
        self.target_key = target_key or config.DEFAULT_TARGET
        self.paths = config.get_target_paths(self.target_key)
        super().__init__(
            name=f"OptimizationAgent[{self.target_key}]",
            config=cfg or config,
        )

    def validate_inputs(self) -> None:
        if not self.paths["features_matrix"].exists():
            raise ValidationError(f"Features de {self.target_key} no encontradas.")
        if not self.paths["modelo_ic50"].exists():
            raise ValidationError(f"Modelo base IC₅₀ de {self.target_key} no encontrado.")

    def execute(self) -> dict:
        df = pd.read_csv(self.paths["features_matrix"])
        feature_cols = [c for c in df.columns if c not in ("pActivity", "canonical_smiles")]
        X = df[feature_cols].values
        y = df["pActivity"].values

        self.logger.info(f"═══ Optimizando {self.target_key} con Optuna ═══")
        opt = optuna_tuner.optimize_hyperparameters(
            X, y, n_trials=self.config.OPTUNA_TRIALS,
            cv_folds=self.config.CV_FOLDS, random_state=self.config.RANDOM_STATE,
        )

        self.logger.info(f"═══ Re-entrenando {self.target_key} con params óptimos ═══")
        optimized = model_trainer.train_model(
            X, y, params=opt["best_params"],
            test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )
        model_trainer.save_model(optimized["model"], self.paths["modelo_ic50_optuna"])

        base = model_trainer.train_model(
            X, y, params=self.config.XGBOOST_DEFAULTS,
            test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )

        return {
            "target": self.target_key,
            "best_params": opt["best_params"],
            "best_rmse": opt["best_rmse"],
            "n_trials": opt["n_trials"],
            "optimized_metrics": optimized["metrics"],
            "base_metrics": base["metrics"],
        }

    def validate_outputs(self, data: dict) -> None:
        if not self.paths["modelo_ic50_optuna"].exists():
            raise ValidationError(f"Modelo optimizado de {self.target_key} no guardado")

    def generate_report(self, data: dict) -> None:
        report_generator.generate_report(
            title=f"⚡ Optimización — {self.target_key}",
            agent_name=self.name,
            sections={
                "Comparación": {
                    "Default R²": data["base_metrics"]["R²"],
                    "Optimizado R²": data["optimized_metrics"]["R²"],
                    "Default RMSE": data["base_metrics"]["RMSE"],
                    "Optimizado RMSE": data["optimized_metrics"]["RMSE"],
                },
                "Mejores Hiperparámetros": {k: str(v) for k, v in data["best_params"].items()},
            },
            output_path=self.paths["reporte_optimizacion"],
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(OptimizationAgent(target_key=target).run())
