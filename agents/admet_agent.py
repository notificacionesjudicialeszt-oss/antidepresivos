"""
ADMETAgent — Agente de Filtrado ADMET (Fase 5).

Soporta múltiples targets (SERT, NAT, DAT).
"""

import logging

import pandas as pd

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import admet_filter, report_generator

logger = logging.getLogger(__name__)


class ADMETAgent(BaseAgent):
    """Agente que aplica filtros farmacológicos para un target."""

    def __init__(self, target_key: str = None, cfg=None):
        self.target_key = target_key or config.DEFAULT_TARGET
        self.paths = config.get_target_paths(self.target_key)
        super().__init__(
            name=f"ADMETAgent[{self.target_key}]",
            config=cfg or config,
        )

    def validate_inputs(self) -> None:
        if not self.paths["dataset_ic50"].exists():
            raise ValidationError(f"Dataset IC₅₀ de {self.target_key} no encontrado.")

    def execute(self) -> dict:
        df = pd.read_csv(self.paths["dataset_ic50"])
        self.logger.info(f"═══ ADMET para {self.target_key}: {len(df)} moléculas ═══")

        df_passed, df_all = admet_filter.full_admet_filter(
            df, smiles_col="canonical_smiles", qed_threshold=self.config.QED_THRESHOLD,
        )

        df_all.to_csv(self.paths["candidatos_evaluados"], index=False)
        df_passed.to_csv(self.paths["candidatos_viables"], index=False)

        return {
            "target": self.target_key,
            "total_evaluated": len(df_all),
            "total_passed": len(df_passed),
            "pass_rate": f"{len(df_passed) / max(len(df_all), 1) * 100:.1f}%",
            "df_passed": df_passed,
            "df_all": df_all,
        }

    def validate_outputs(self, data: dict) -> None:
        if data["total_evaluated"] == 0:
            raise ValidationError(f"Ninguna molécula evaluada para {self.target_key}")
        if data["total_passed"] == 0:
            self.logger.warning(f"⚠️  {self.target_key}: ningún candidato pasó todos los filtros")

    def generate_report(self, data: dict) -> None:
        top = []
        if data["total_passed"] > 0:
            for _, row in data["df_passed"].head(10).iterrows():
                top.append(
                    f"**{row.get('molecule_chembl_id', 'N/A')}** — "
                    f"QED: {row.get('QED', 'N/A')}, pActivity: {row.get('pActivity', 0):.2f}"
                )

        report_generator.generate_report(
            title=f"💊 ADMET — {self.target_key}",
            agent_name=self.name,
            sections={
                "Resumen": {
                    "Evaluados": data["total_evaluated"],
                    "Viables": data["total_passed"],
                    "Tasa": data["pass_rate"],
                },
                "Top 10 Candidatos": top or ["Ninguno pasó todos los filtros"],
            },
            output_path=self.paths["reporte_admet"],
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(ADMETAgent(target_key=target).run())
