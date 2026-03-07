"""
FeatureAgent — Agente de Feature Engineering (Fase 2).

Responsabilidades:
    1. Calcular fingerprints ECFP4 y filtrar por varianza
    2. Calcular los 43 descriptores fisicoquímicos
    3. Concatenar en la matriz combinada de ~887 features
    4. Normalizar descriptores numéricos
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import (
    descriptor_calculator,
    fingerprint_calculator,
    report_generator,
)

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """Agente que construye la matriz de features combinada."""

    def __init__(self, cfg=None):
        super().__init__(name="FeatureAgent", config=cfg or config)

    def validate_inputs(self) -> None:
        """Verifica que los datasets limpios existen."""
        if not self.config.DATASET_IC50.exists():
            raise ValidationError(
                f"Dataset IC₅₀ no encontrado: {self.config.DATASET_IC50}\n"
                f"Ejecuta primero el DataAgent."
            )

    def execute(self) -> dict:
        """
        1. Carga dataset IC₅₀
        2. Calcula fingerprints + filtra por varianza
        3. Calcula 43 descriptores
        4. Concatena y normaliza
        """
        # Cargar dataset
        df = pd.read_csv(self.config.DATASET_IC50)
        smiles_list = df["canonical_smiles"].tolist()
        y = df["pActivity"].values

        self.logger.info(f"📂 Dataset cargado: {len(df)} moléculas")

        # ── Skill 1: Fingerprints ──
        self.logger.info("═══ Calculando ECFP4 fingerprints ═══")
        fp_matrix = fingerprint_calculator.calculate_fingerprints(
            smiles_list,
            n_bits=self.config.FINGERPRINT_BITS,
            radius=self.config.FINGERPRINT_RADIUS,
        )

        fp_filtered, retained_bits = fingerprint_calculator.filter_by_variance(
            fp_matrix,
            threshold=self.config.VARIANCE_THRESHOLD,
        )

        # ── Skill 2: Descriptores ──
        self.logger.info("═══ Calculando 43 descriptores ═══")
        desc_df = descriptor_calculator.calculate_descriptors_batch(smiles_list)

        # ── Normalizar descriptores ──
        self.logger.info("⚖️  Normalizando descriptores con StandardScaler...")
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_df.values)

        # ── Concatenar: fingerprints_filtrados + descriptores_normalizados ──
        X_combined = np.hstack([fp_filtered, desc_scaled])
        self.logger.info(
            f"🔗 Matriz final: {X_combined.shape[0]} moléculas × {X_combined.shape[1]} features "
            f"({fp_filtered.shape[1]} bits + {desc_scaled.shape[1]} descriptores)"
        )

        # Guardar como CSV
        feature_names = (
            [f"FP_bit_{i}" for i in retained_bits]
            + list(desc_df.columns)
        )
        df_features = pd.DataFrame(X_combined, columns=feature_names)
        df_features["pActivity"] = y
        df_features["canonical_smiles"] = smiles_list
        df_features.to_csv(self.config.FEATURES_MATRIX, index=False)

        return {
            "X": X_combined,
            "y": y,
            "n_fp_bits": fp_filtered.shape[1],
            "n_descriptors": desc_scaled.shape[1],
            "n_total_features": X_combined.shape[1],
            "n_molecules": X_combined.shape[0],
            "feature_names": feature_names,
            "scaler": scaler,
        }

    def validate_outputs(self, data: dict) -> None:
        """Verifica que la matriz de features es coherente."""
        if data["n_molecules"] == 0:
            raise ValidationError("La matriz de features está vacía")
        if data["n_total_features"] < 50:
            raise ValidationError(
                f"Muy pocas features ({data['n_total_features']}). "
                f"Algo salió mal con los fingerprints o descriptores."
            )
        if not self.config.FEATURES_MATRIX.exists():
            raise ValidationError(f"Archivo no guardado: {self.config.FEATURES_MATRIX}")

    def generate_report(self, data: dict) -> None:
        report_generator.generate_report(
            title="🧬 Reporte de Feature Engineering — Fase 2",
            agent_name=self.name,
            sections={
                "Resumen": {
                    "Moléculas": data["n_molecules"],
                    "Bits ECFP4 (post-filtro)": data["n_fp_bits"],
                    "Descriptores fisicoquímicos": data["n_descriptors"],
                    "Total features": data["n_total_features"],
                },
                "Archivos Generados": [
                    f"`{self.config.FEATURES_MATRIX}`",
                ],
            },
            output_path=self.config.REPORTES_DIR / "reporte_features.md",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = FeatureAgent()
    result = agent.run()
    print(result)
