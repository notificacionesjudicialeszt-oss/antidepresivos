"""
ADMETAgent — Agente de Filtrado ADMET (Fase 5).

Responsabilidades:
    1. Tomar los candidatos del modelo IC₅₀
    2. Evaluar filtros ADMET: QED, Lipinski, BBB, PAINS
    3. Generar tabla final de candidatos viables
    4. Exportar resultados
"""

import logging

import pandas as pd

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import admet_filter, report_generator

logger = logging.getLogger(__name__)


class ADMETAgent(BaseAgent):
    """Agente que aplica filtros farmacológicos a los candidatos."""

    def __init__(self, cfg=None):
        super().__init__(name="ADMETAgent", config=cfg or config)

    def validate_inputs(self) -> None:
        if not self.config.DATASET_IC50.exists():
            raise ValidationError("Dataset IC₅₀ no encontrado.")

    def execute(self) -> dict:
        """
        Aplica el pipeline completo de filtros ADMET a todas las moléculas
        del dataset IC₅₀ curado.
        """

        # Cargar moléculas
        df = pd.read_csv(self.config.DATASET_IC50)
        self.logger.info(f"📂 {len(df)} moléculas cargadas para evaluación ADMET")

        # ── Skill: Filtros ADMET ──
        self.logger.info("═══ Aplicando filtros ADMET ═══")
        df_passed, df_all = admet_filter.full_admet_filter(
            df,
            smiles_col="canonical_smiles",
            qed_threshold=self.config.QED_THRESHOLD,
        )

        # Guardar resultados
        output_all = self.config.REPORTES_DIR / "candidatos_evaluados.csv"
        output_passed = self.config.REPORTES_DIR / "candidatos_viables.csv"

        df_all.to_csv(output_all, index=False)
        df_passed.to_csv(output_passed, index=False)

        self.logger.info(f"💾 Todos los candidatos: {output_all}")
        self.logger.info(f"💾 Candidatos viables: {output_passed}")

        return {
            "total_evaluated": len(df_all),
            "total_passed": len(df_passed),
            "pass_rate": f"{len(df_passed) / max(len(df_all), 1) * 100:.1f}%",
            "df_passed": df_passed,
            "df_all": df_all,
            "output_all": str(output_all),
            "output_passed": str(output_passed),
        }

    def validate_outputs(self, data: dict) -> None:
        if data["total_evaluated"] == 0:
            raise ValidationError("No se evaluó ninguna molécula")

        if data["total_passed"] == 0:
            self.logger.warning(
                "⚠️  ¡Ninguna molécula pasó todos los filtros ADMET! "
                "Considera relajar los umbrales."
            )

    def generate_report(self, data: dict) -> None:
        # Top 10 candidatos viables
        top_candidates = []
        if data["total_passed"] > 0:
            top = data["df_passed"].head(10)
            for _, row in top.iterrows():
                top_candidates.append(
                    f"**{row.get('molecule_chembl_id', 'N/A')}** — "
                    f"QED: {row.get('QED', 'N/A')}, "
                    f"BBB: {row.get('BBB_score', 'N/A')}, "
                    f"pActivity: {row.get('pActivity', 'N/A'):.2f}"
                )

        report_generator.generate_report(
            title="💊 Reporte ADMET — Fase 5",
            agent_name=self.name,
            sections={
                "Resumen": {
                    "Moléculas evaluadas": data["total_evaluated"],
                    "Pasaron todos los filtros": data["total_passed"],
                    "Tasa de aprobación": data["pass_rate"],
                },
                "Top 10 Candidatos Viables": top_candidates if top_candidates else ["Ningún candidato pasó todos los filtros"],
                "Archivos Generados": [
                    f"`{data['output_all']}`",
                    f"`{data['output_passed']}`",
                ],
            },
            output_path=self.config.REPORTES_DIR / "reporte_admet.md",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = ADMETAgent()
    result = agent.run()
    print(result)
