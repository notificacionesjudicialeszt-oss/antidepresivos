"""
TrainingAgent — Agente de Entrenamiento Dual-Endpoint (Fase 3).

Responsabilidades:
    1. Entrenar modelo IC₅₀ (eficacia bioquímica)
    2. Entrenar modelo EC₅₀ (eficacia celular / toxicidad)
    3. Validar independencia estadística entre ambos
    4. Generar scatter plots y métricas comparativas
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import model_trainer, report_generator

logger = logging.getLogger(__name__)


class TrainingAgent(BaseAgent):
    """Agente que entrena los modelos duales IC₅₀ y EC₅₀."""

    def __init__(self, cfg=None):
        super().__init__(name="TrainingAgent", config=cfg or config)

    def validate_inputs(self) -> None:
        if not self.config.FEATURES_MATRIX.exists():
            raise ValidationError(
                f"Matriz de features no encontrada: {self.config.FEATURES_MATRIX}\n"
                f"Ejecuta primero el FeatureAgent."
            )
        if not self.config.DATASET_IC50.exists():
            raise ValidationError(f"Dataset IC₅₀ no encontrado.")

    def execute(self) -> dict:
        """Entrena modelos para IC₅₀ y opcionalmente EC₅₀."""

        # ── Cargar features ──
        df_features = pd.read_csv(self.config.FEATURES_MATRIX)
        feature_cols = [c for c in df_features.columns
                        if c not in ("pActivity", "canonical_smiles")]
        X = df_features[feature_cols].values
        y_ic50 = df_features["pActivity"].values

        # ── Modelo IC₅₀ ──
        self.logger.info("═══ Entrenando modelo IC₅₀ ═══")
        ic50_result = model_trainer.train_model(
            X, y_ic50,
            params=self.config.XGBOOST_DEFAULTS,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            cv_folds=self.config.CV_FOLDS,
        )
        model_trainer.save_model(ic50_result["model"], self.config.MODELO_IC50)

        result = {
            "ic50_metrics": ic50_result["metrics"],
            "ic50_splits": ic50_result["splits"],
            "has_ec50": False,
        }

        # ── Modelo EC₅₀ (si hay datos suficientes) ──
        if self.config.DATASET_EC50.exists():
            df_ec50 = pd.read_csv(self.config.DATASET_EC50)
            if len(df_ec50) >= 50:
                self.logger.info(f"═══ Entrenando modelo EC₅₀ ({len(df_ec50)} registros) ═══")

                # Necesitamos recalcular features para EC₅₀
                from skills import fingerprint_calculator, descriptor_calculator
                from sklearn.preprocessing import StandardScaler

                smiles_ec50 = df_ec50["canonical_smiles"].tolist()
                y_ec50 = df_ec50["pActivity"].values

                fp_ec50 = fingerprint_calculator.calculate_fingerprints(
                    smiles_ec50,
                    n_bits=self.config.FINGERPRINT_BITS,
                    radius=self.config.FINGERPRINT_RADIUS,
                )
                fp_ec50_filt, _ = fingerprint_calculator.filter_by_variance(
                    fp_ec50, threshold=self.config.VARIANCE_THRESHOLD
                )
                desc_ec50 = descriptor_calculator.calculate_descriptors_batch(smiles_ec50)
                scaler = StandardScaler()
                desc_ec50_scaled = scaler.fit_transform(desc_ec50.values)
                X_ec50 = np.hstack([fp_ec50_filt, desc_ec50_scaled])

                ec50_result = model_trainer.train_model(
                    X_ec50, y_ec50,
                    params=self.config.XGBOOST_DEFAULTS,
                    test_size=self.config.TEST_SIZE,
                    random_state=self.config.RANDOM_STATE,
                    cv_folds=self.config.CV_FOLDS,
                )
                model_trainer.save_model(ec50_result["model"], self.config.MODELO_EC50)

                result["ec50_metrics"] = ec50_result["metrics"]
                result["ec50_splits"] = ec50_result["splits"]
                result["has_ec50"] = True
            else:
                self.logger.warning(
                    f"⚠️  EC₅₀ tiene solo {len(df_ec50)} registros. "
                    f"Se requieren mínimo 50 para entrenar un modelo. Saltando."
                )

        return result

    def validate_outputs(self, data: dict) -> None:
        if not self.config.MODELO_IC50.exists():
            raise ValidationError("Modelo IC₅₀ no fue guardado")

        r2 = float(data["ic50_metrics"]["R²"])
        if r2 < 0.0:
            raise ValidationError(
                f"R² del modelo IC₅₀ es negativo ({r2:.4f}). "
                f"El modelo es peor que predecir la media."
            )

    def generate_report(self, data: dict) -> None:
        sections = {
            "Modelo IC₅₀ (Eficacia Bioquímica)": data["ic50_metrics"],
        }

        if data["has_ec50"]:
            sections["Modelo EC₅₀ (Eficacia Celular)"] = data["ec50_metrics"]
        else:
            sections["Modelo EC₅₀"] = "⚠️  No se entrenó (datos insuficientes)"

        sections["Modelos Guardados"] = [
            f"`{self.config.MODELO_IC50}`",
        ]
        if data["has_ec50"]:
            sections["Modelos Guardados"].append(f"`{self.config.MODELO_EC50}`")

        report_generator.generate_report(
            title="🧠 Reporte de Entrenamiento — Fase 3",
            agent_name=self.name,
            sections=sections,
            output_path=self.config.REPORTES_DIR / "reporte_entrenamiento.md",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = TrainingAgent()
    result = agent.run()
    print(result)
