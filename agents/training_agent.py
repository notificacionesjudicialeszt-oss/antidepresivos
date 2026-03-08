"""
TrainingAgent — Agente de Entrenamiento Dual-Endpoint (Fase 3).

Soporta múltiples targets (SERT, NAT, DAT).
"""

import logging

import numpy as np
import pandas as pd

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import model_trainer, report_generator

logger = logging.getLogger(__name__)


class TrainingAgent(BaseAgent):
    """Agente que entrena modelos duales IC₅₀ / EC₅₀ para un target."""

    def __init__(self, target_key: str = None, cfg=None):
        self.target_key = target_key or config.DEFAULT_TARGET
        self.paths = config.get_target_paths(self.target_key)
        super().__init__(
            name=f"TrainingAgent[{self.target_key}]",
            config=cfg or config,
        )

    def validate_inputs(self) -> None:
        if not self.paths["features_matrix"].exists():
            raise ValidationError(
                f"Features de {self.target_key} no encontradas. Ejecuta Fase 2 primero."
            )

    def execute(self) -> dict:
        df = pd.read_csv(self.paths["features_matrix"])
        feature_cols = [c for c in df.columns if c not in ("pActivity", "canonical_smiles")]
        X = df[feature_cols].values
        y = df["pActivity"].values

        # ── Modelo IC₅₀ ──
        self.logger.info(f"═══ Entrenando {self.target_key} IC₅₀ ═══")
        ic50_result = model_trainer.train_model(
            X, y, params=self.config.XGBOOST_DEFAULTS,
            test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )
        model_trainer.save_model(ic50_result["model"], self.paths["modelo_ic50"])

        result = {"target": self.target_key, "ic50_metrics": ic50_result["metrics"], "has_ec50": False}

        # ── Modelo EC₅₀ si hay datos ──
        if self.paths["dataset_ec50"].exists():
            df_ec50 = pd.read_csv(self.paths["dataset_ec50"])
            if len(df_ec50) >= 50:
                self.logger.info(f"═══ Entrenando {self.target_key} EC₅₀ ({len(df_ec50)} reg) ═══")
                from skills import fingerprint_calculator, descriptor_calculator
                from sklearn.preprocessing import StandardScaler

                smiles_ec50 = df_ec50["canonical_smiles"].tolist()
                y_ec50 = df_ec50["pActivity"].values
                fp = fingerprint_calculator.calculate_fingerprints(smiles_ec50, n_bits=self.config.FINGERPRINT_BITS, radius=self.config.FINGERPRINT_RADIUS)
                fp_filt, _ = fingerprint_calculator.filter_by_variance(fp, threshold=self.config.VARIANCE_THRESHOLD)
                desc = descriptor_calculator.calculate_descriptors_batch(smiles_ec50)
                X_ec50 = np.hstack([fp_filt, StandardScaler().fit_transform(desc.values)])

                ec50_result = model_trainer.train_model(
                    X_ec50, y_ec50, params=self.config.XGBOOST_DEFAULTS,
                    test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE,
                    cv_folds=self.config.CV_FOLDS,
                )
                model_trainer.save_model(ec50_result["model"], self.paths["modelo_ec50"])
                result["ec50_metrics"] = ec50_result["metrics"]
                result["has_ec50"] = True

        return result

    def validate_outputs(self, data: dict) -> None:
        if not self.paths["modelo_ic50"].exists():
            raise ValidationError(f"Modelo IC₅₀ de {self.target_key} no guardado")

    def generate_report(self, data: dict) -> None:
        sections = {"Modelo IC₅₀": data["ic50_metrics"]}
        if data["has_ec50"]:
            sections["Modelo EC₅₀"] = data["ec50_metrics"]
        report_generator.generate_report(
            title=f"🧠 Entrenamiento — {self.target_key}",
            agent_name=self.name, sections=sections,
            output_path=self.paths["reporte_entrenamiento"],
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(TrainingAgent(target_key=target).run())
