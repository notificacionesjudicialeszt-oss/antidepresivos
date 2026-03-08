"""
DataAgent — Agente de Adquisición y Curación de Datos (Fase 1).

Soporta múltiples targets (SERT, NAT, DAT).
"""

import logging

import config
from agents.base_agent import BaseAgent, ValidationError
from skills import chembl_scraper, smiles_validator, report_generator

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Agente que descarga y cura datos de ChEMBL para un target dado."""

    def __init__(self, target_key: str = None, cfg=None):
        self.target_key = target_key or config.DEFAULT_TARGET
        self.target_info = config.TARGETS[self.target_key]
        self.paths = config.get_target_paths(self.target_key)
        super().__init__(
            name=f"DataAgent[{self.target_key}]",
            config=cfg or config,
        )

    def validate_inputs(self) -> None:
        if self.target_key not in config.TARGETS:
            raise ValidationError(f"Target '{self.target_key}' no existe. Usa: {list(config.TARGETS.keys())}")
        self.logger.info(
            f"🎯 Target: {self.target_info['chembl_id']} — "
            f"{self.target_info['name']} ({self.target_info['neurotransmitter']})"
        )

    def execute(self) -> dict:
        chembl_id = self.target_info["chembl_id"]

        # ── Scraping ──
        self.logger.info(f"═══ Scraping {self.target_key} desde ChEMBL ═══")
        df_crudo = chembl_scraper.scrape_target(
            target_id=chembl_id,
            base_url=self.config.API_BASE_URL,
            limit=self.config.LIMIT_PER_PAGE,
        )
        self.config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        df_crudo.to_csv(self.paths["dataset_crudo"], index=False)

        # ── Curación IC₅₀ ──
        self.logger.info(f"═══ Curación IC₅₀ para {self.target_key} ═══")
        df_ic50 = smiles_validator.curate_dataset(df_crudo, standard_type_filter="IC50")
        df_ic50.to_csv(self.paths["dataset_ic50"], index=False)
        ic50_stats = smiles_validator.generate_curation_stats(len(df_crudo), df_ic50)
        self.logger.info(f"💾 {self.target_key} IC₅₀: {len(df_ic50):,} registros")

        # ── Curación EC₅₀ ──
        self.logger.info(f"═══ Curación EC₅₀ para {self.target_key} ═══")
        df_ec50 = smiles_validator.curate_dataset(df_crudo, standard_type_filter="EC50")
        df_ec50.to_csv(self.paths["dataset_ec50"], index=False)
        ec50_stats = smiles_validator.generate_curation_stats(len(df_crudo), df_ec50)
        self.logger.info(f"💾 {self.target_key} EC₅₀: {len(df_ec50):,} registros")

        return {
            "target": self.target_key,
            "df_crudo_count": len(df_crudo),
            "df_ic50": df_ic50,
            "df_ec50": df_ec50,
            "ic50_stats": ic50_stats,
            "ec50_stats": ec50_stats,
        }

    def validate_outputs(self, data: dict) -> None:
        ic50_count = len(data["df_ic50"])
        if ic50_count == 0:
            raise ValidationError(f"❌ {self.target_key} IC₅₀ vacío después de curación")
        if ic50_count < self.config.MIN_DATASET_SIZE:
            self.logger.warning(
                f"⚠️  {self.target_key} IC₅₀: {ic50_count:,} registros "
                f"(mínimo deseado: {self.config.MIN_DATASET_SIZE:,})"
            )
        else:
            self.logger.info(f"✅ {self.target_key} IC₅₀: {ic50_count:,} registros")

    def generate_report(self, data: dict) -> None:
        report_generator.generate_report(
            title=f"📊 Curación de Datos — {self.target_key}",
            agent_name=self.name,
            sections={
                "Target": f"**{self.target_info['name']}** ({self.target_info['chembl_id']})",
                "Neurotransmisor": self.target_info["neurotransmitter"],
                "Dataset Crudo": f"{data['df_crudo_count']:,} registros",
                "Curación IC₅₀": data["ic50_stats"],
                "Curación EC₅₀": data["ec50_stats"],
            },
            output_path=self.paths["reporte_curacion"],
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    agent = DataAgent(target_key=target)
    print(agent.run())
