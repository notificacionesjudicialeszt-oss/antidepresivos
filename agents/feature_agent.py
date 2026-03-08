"""
FeatureAgent — Agente de Feature Engineering (Fase 2).

Soporta múltiples targets (SERT, NAT, DAT).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import descriptor_calculator, fingerprint_calculator, report_generator

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """Agente que construye la matriz de features combinada para un target."""

    def __init__(self, target_key: str = None, cfg=None):
        self.target_key = target_key or config.DEFAULT_TARGET
        self.paths = config.get_target_paths(self.target_key)
        super().__init__(
            name=f"FeatureAgent[{self.target_key}]",
            config=cfg or config,
        )

    def validate_inputs(self) -> None:
        if not self.paths["dataset_ic50"].exists():
            raise ValidationError(
                f"Dataset IC₅₀ de {self.target_key} no encontrado: {self.paths['dataset_ic50']}\n"
                f"Ejecuta primero: python main.py --fase 1 --target {self.target_key}"
            )

    def execute(self) -> dict:
        df = pd.read_csv(self.paths["dataset_ic50"])
        smiles_list = df["canonical_smiles"].tolist()
        y = df["pActivity"].values

        self.logger.info(f"📂 {self.target_key}: {len(df)} moléculas cargadas")

        # ── Fingerprints ──
        self.logger.info(f"═══ ECFP4 para {self.target_key} ═══")
        fp_matrix = fingerprint_calculator.calculate_fingerprints(
            smiles_list, n_bits=self.config.FINGERPRINT_BITS, radius=self.config.FINGERPRINT_RADIUS,
        )
        fp_filtered, retained_bits = fingerprint_calculator.filter_by_variance(
            fp_matrix, threshold=self.config.VARIANCE_THRESHOLD,
        )

        # ── Descriptores ──
        self.logger.info(f"═══ 43 descriptores para {self.target_key} ═══")
        desc_df = descriptor_calculator.calculate_descriptors_batch(smiles_list)

        # ── Normalizar + concatenar ──
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_df.values)
        X_combined = np.hstack([fp_filtered, desc_scaled])

        self.logger.info(
            f"🔗 {self.target_key}: {X_combined.shape[0]} mol × {X_combined.shape[1]} features "
            f"({fp_filtered.shape[1]} bits + {desc_scaled.shape[1]} desc)"
        )

        # Guardar
        feature_names = [f"FP_bit_{i}" for i in retained_bits] + list(desc_df.columns)
        df_features = pd.DataFrame(X_combined, columns=feature_names)
        df_features["pActivity"] = y
        df_features["canonical_smiles"] = smiles_list
        df_features.to_csv(self.paths["features_matrix"], index=False)

        return {
            "target": self.target_key,
            "X": X_combined, "y": y,
            "n_fp_bits": fp_filtered.shape[1],
            "n_descriptors": desc_scaled.shape[1],
            "n_total_features": X_combined.shape[1],
            "n_molecules": X_combined.shape[0],
        }

    def validate_outputs(self, data: dict) -> None:
        if data["n_molecules"] == 0:
            raise ValidationError(f"{self.target_key}: matriz de features vacía")
        if data["n_total_features"] < 50:
            raise ValidationError(f"{self.target_key}: muy pocas features ({data['n_total_features']})")

    def generate_report(self, data: dict) -> None:
        report_generator.generate_report(
            title=f"🧬 Feature Engineering — {self.target_key}",
            agent_name=self.name,
            sections={
                "Resumen": {
                    "Moléculas": data["n_molecules"],
                    "Bits ECFP4": data["n_fp_bits"],
                    "Descriptores": data["n_descriptors"],
                    "Total features": data["n_total_features"],
                },
            },
            output_path=self.paths["reporte_features"],
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(FeatureAgent(target_key=target).run())
