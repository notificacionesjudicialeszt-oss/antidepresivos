"""
Skill: SMILES Validator
Validación, canonicalización y curación de datos moleculares.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> Optional[str]:
    """
    Valida un SMILES y retorna su forma canónica.
    Retorna None si es inválido.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def curate_dataset(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    pchembl_col: str = "pchembl_value",
    standard_type_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cura un dataset de actividades de ChEMBL.

    Pasos:
        1. Filtra por standard_type si se especifica (ej: "IC50", "EC50")
        2. Valida cada SMILES con RDKit
        3. Convierte pchembl_value a float
        4. Elimina registros con data_validity_comment problemático
        5. Elimina duplicados por SMILES canónico (mediana de pActivity)
        6. Elimina outliers (pActivity fuera de [3, 12])

    Args:
        df: DataFrame crudo de ChEMBL.
        smiles_col: Nombre de la columna de SMILES.
        pchembl_col: Nombre de la columna de pChEMBL value.
        standard_type_filter: Filtrar por tipo (ej: "IC50").

    Returns:
        DataFrame curado con columnas: canonical_smiles, pActivity, y metadatos.
    """
    initial_count = len(df)
    logger.info(f"📊 Iniciando curación: {initial_count:,} registros")

    # 1. Filtrar por standard_type
    if standard_type_filter:
        df = df[df["standard_type"].str.upper() == standard_type_filter.upper()].copy()
        logger.info(f"  🔬 Filtro {standard_type_filter}: {len(df):,} registros")

    # 2. Eliminar registros con comentarios de validez problemáticos
    problematic_comments = ["Outside typical range", "Potential transcription error", "Potential author error"]
    if "data_validity_comment" in df.columns:
        mask = df["data_validity_comment"].isna() | ~df["data_validity_comment"].isin(problematic_comments)
        df = df[mask].copy()
        logger.info(f"  🧹 Post-validez: {len(df):,} registros")

    # 3. Eliminar filas sin SMILES o sin pActivity
    df = df.dropna(subset=[smiles_col, pchembl_col]).copy()

    # 4. Convertir pchembl_value a float
    df["pActivity"] = pd.to_numeric(df[pchembl_col], errors="coerce")
    df = df.dropna(subset=["pActivity"])

    # 5. Validar SMILES con RDKit
    logger.info("  🧪 Validando SMILES con RDKit...")
    df["validated_smiles"] = df[smiles_col].apply(validate_smiles)
    invalid_count = df["validated_smiles"].isna().sum()
    df = df.dropna(subset=["validated_smiles"]).copy()
    logger.info(f"  ❌ SMILES inválidos descartados: {invalid_count}")

    # 6. Eliminar outliers de pActivity (rango biológicamente razonable)
    df = df[(df["pActivity"] >= 3.0) & (df["pActivity"] <= 12.0)].copy()
    logger.info(f"  📐 Post-filtro outliers (pActivity 3-12): {len(df):,} registros")

    # 7. Deduplicar por SMILES canónico — quedarse con la mediana
    df_dedup = (
        df.groupby("validated_smiles")
        .agg(
            pActivity=("pActivity", "median"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
            n_measurements=("pActivity", "count"),
        )
        .reset_index()
        .rename(columns={"validated_smiles": "canonical_smiles"})
    )

    logger.info(
        f"✅ Curación completa: {initial_count:,} → {len(df_dedup):,} registros "
        f"({initial_count - len(df_dedup):,} eliminados)"
    )

    return df_dedup


def generate_curation_stats(
    original_count: int,
    curated_df: pd.DataFrame,
) -> dict:
    """Genera estadísticas de curación para el reporte."""
    return {
        "registros_originales": original_count,
        "registros_finales": len(curated_df),
        "eliminados": original_count - len(curated_df),
        "porcentaje_retenido": f"{len(curated_df) / max(original_count, 1) * 100:.1f}%",
        "pActivity_mean": f"{curated_df['pActivity'].mean():.2f}",
        "pActivity_std": f"{curated_df['pActivity'].std():.2f}",
        "pActivity_min": f"{curated_df['pActivity'].min():.2f}",
        "pActivity_max": f"{curated_df['pActivity'].max():.2f}",
        "moleculas_unicas": len(curated_df),
    }
