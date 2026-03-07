"""
Skill: ChEMBL Scraper
Descarga datos de actividad biológica desde la API de ChEMBL con paginación automática.
Reutilizable para cualquier target.
"""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def scrape_target(
    target_id: str,
    base_url: str = "https://www.ebi.ac.uk/chembl/api/data",
    limit: int = 500,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Descarga TODOS los registros de actividad biológica para un target de ChEMBL.

    Args:
        target_id: ID del target en ChEMBL (ej: "CHEMBL228" para SERT).
        base_url: URL base de la API.
        limit: Registros por página (máx 1000).
        max_retries: Reintentos por página en caso de error.
        retry_delay: Segundos entre reintentos.

    Returns:
        DataFrame con columnas: molecule_chembl_id, canonical_smiles,
        pchembl_value, standard_type, standard_value, standard_units,
        assay_type, data_validity_comment, assay_description.
    """
    all_activities = []
    offset = 0
    total = None
    page = 1

    logger.info(f"🌐 Iniciando scraping de {target_id} desde ChEMBL...")

    while True:
        url = (
            f"{base_url}/activity.json"
            f"?target_chembl_id={target_id}"
            f"&pchembl_value__isnull=false"
            f"&limit={limit}"
            f"&offset={offset}"
        )

        # Retry con backoff
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(
                    f"⚠️  Intento {attempt}/{max_retries} falló para offset={offset}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                else:
                    raise RuntimeError(
                        f"No se pudo descargar offset={offset} después de {max_retries} intentos"
                    )

        data = response.json()

        # En la primera página, capturamos el total
        if total is None:
            total = data["page_meta"]["total_count"]
            total_pages = (total + limit - 1) // limit
            logger.info(f"📊 Total de registros disponibles: {total:,} ({total_pages} páginas)")

        # Extraer actividades de esta página
        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            all_activities.append({
                "molecule_chembl_id": act.get("molecule_chembl_id"),
                "canonical_smiles": act.get("canonical_smiles"),
                "pchembl_value": act.get("pchembl_value"),
                "standard_type": act.get("standard_type"),
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units"),
                "assay_type": act.get("assay_type"),
                "data_validity_comment": act.get("data_validity_comment"),
                "assay_description": act.get("assay_description"),
            })

        logger.info(f"  📄 Página {page}/{total_pages} — {len(all_activities):,} registros acumulados")

        # ¿Hay más páginas?
        if data["page_meta"].get("next") is None:
            break

        offset += limit
        page += 1
        time.sleep(0.3)  # Rate limiting cortés

    df = pd.DataFrame(all_activities)
    logger.info(f"✅ Scraping completo: {len(df):,} registros descargados para {target_id}")

    return df
