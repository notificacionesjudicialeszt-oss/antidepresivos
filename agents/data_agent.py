"""
DataAgent — Agente de Adquisición y Curación de Datos (Fase 1).

Responsabilidades:
    1. Descargar TODOS los registros de actividad biológica del SERT desde ChEMBL
    2. Curar y limpiar los datos (validar SMILES, eliminar duplicados, etc.)
    3. Separar en datasets IC₅₀ y EC₅₀
    4. Generar reporte de curación
"""

import logging

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import chembl_scraper, smiles_validator, report_generator

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Agente que descarga y cura datos de ChEMBL para un target dado."""

    def __init__(self, cfg=None):
        super().__init__(name="DataAgent", config=cfg or config)

    def validate_inputs(self) -> None:
        """Verifica que la configuración tiene lo necesario."""
        if not self.config.TARGET_ID:
            raise ValidationError("TARGET_ID no está definido en config.py")
        if not self.config.API_BASE_URL:
            raise ValidationError("API_BASE_URL no está definido en config.py")
        self.logger.info(f"🎯 Target: {self.config.TARGET_ID} ({self.config.TARGET_NAME})")

    def execute(self) -> dict:
        """
        1. Scrapea ChEMBL para el target configurado
        2. Cura los datos para IC₅₀ y EC₅₀ por separado
        3. Guarda los datasets limpios
        """

        # ── Skill 1: Scraping ──
        self.logger.info("═══ Fase 1.1: Scraping de ChEMBL ═══")
        df_crudo = chembl_scraper.scrape_target(
            target_id=self.config.TARGET_ID,
            base_url=self.config.API_BASE_URL,
            limit=self.config.LIMIT_PER_PAGE,
        )

        # Guardar dataset crudo
        self.config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        df_crudo.to_csv(self.config.DATASET_CRUDO, index=False)
        self.logger.info(f"💾 Dataset crudo guardado: {self.config.DATASET_CRUDO}")

        # ── Skill 2: Curación IC₅₀ ──
        self.logger.info("═══ Fase 1.2: Curación de datos IC₅₀ ═══")
        df_ic50 = smiles_validator.curate_dataset(
            df_crudo,
            standard_type_filter="IC50",
        )
        df_ic50.to_csv(self.config.DATASET_IC50, index=False)
        ic50_stats = smiles_validator.generate_curation_stats(len(df_crudo), df_ic50)
        self.logger.info(f"💾 Dataset IC₅₀: {len(df_ic50):,} registros → {self.config.DATASET_IC50}")

        # ── Skill 2: Curación EC₅₀ ──
        self.logger.info("═══ Fase 1.3: Curación de datos EC₅₀ ═══")
        df_ec50 = smiles_validator.curate_dataset(
            df_crudo,
            standard_type_filter="EC50",
        )
        df_ec50.to_csv(self.config.DATASET_EC50, index=False)
        ec50_stats = smiles_validator.generate_curation_stats(len(df_crudo), df_ec50)
        self.logger.info(f"💾 Dataset EC₅₀: {len(df_ec50):,} registros → {self.config.DATASET_EC50}")

        return {
            "df_crudo_count": len(df_crudo),
            "df_ic50": df_ic50,
            "df_ec50": df_ec50,
            "ic50_stats": ic50_stats,
            "ec50_stats": ec50_stats,
        }

    def validate_outputs(self, data: dict) -> None:
        """Verifica que los datasets cumplen los criterios mínimos."""
        ic50_count = len(data["df_ic50"])

        if ic50_count == 0:
            raise ValidationError("❌ El dataset IC₅₀ está vacío después de la curación")

        if ic50_count < self.config.MIN_DATASET_SIZE:
            self.logger.warning(
                f"⚠️  IC₅₀ tiene {ic50_count:,} registros, "
                f"por debajo del mínimo deseado ({self.config.MIN_DATASET_SIZE:,}). "
                f"Se continuará pero los resultados pueden ser limitados."
            )
        else:
            self.logger.info(
                f"✅ IC₅₀: {ic50_count:,} registros (meta: {self.config.MIN_DATASET_SIZE:,}) ¡Cumplido!"
            )

        ec50_count = len(data["df_ec50"])
        if ec50_count < 50:
            self.logger.warning(
                f"⚠️  EC₅₀ tiene solo {ec50_count} registros. "
                f"Puede no ser suficiente para un modelo independiente."
            )

    def generate_report(self, data: dict) -> None:
        """Genera reporte de curación."""
        report_generator.generate_report(
            title="📊 Reporte de Curación de Datos — Fase 1",
            agent_name=self.name,
            sections={
                "Target": f"**{self.config.TARGET_NAME}** ({self.config.TARGET_ID})",
                "Dataset Crudo": f"{data['df_crudo_count']:,} registros descargados de ChEMBL",
                "Curación IC₅₀": data["ic50_stats"],
                "Curación EC₅₀": data["ec50_stats"],
                "Archivos Generados": [
                    f"`{self.config.DATASET_CRUDO}`",
                    f"`{self.config.DATASET_IC50}`",
                    f"`{self.config.DATASET_EC50}`",
                ],
            },
            output_path=self.config.REPORTES_DIR / "reporte_curacion.md",
        )


# ── Para ejecutar solo este agente ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DataAgent()
    result = agent.run()
    print(result)
