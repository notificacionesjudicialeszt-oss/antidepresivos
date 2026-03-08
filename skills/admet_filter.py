"""
Skill: ADMET Filter
Filtros farmacológicos: QED, Lipinski, BBB, PAINS.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

logger = logging.getLogger(__name__)


def calculate_qed(smiles: str) -> float:
    """Calcula el QED (Quantitative Estimate of Drug-likeness) de una molécula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return Descriptors.qed(mol)


def check_lipinski(smiles: str) -> dict:
    """
    Evalúa la Regla de 5 de Lipinski.

    Returns:
        Dict con cada regla (True = pasa) y el conteo de violaciones.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"passes": False, "violations": 5, "rules": {}}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    rules = {
        "MW ≤ 500": mw <= 500,
        "LogP ≤ 5": logp <= 5,
        "HBD ≤ 5": hbd <= 5,
        "HBA ≤ 10": hba <= 10,
    }

    violations = sum(1 for v in rules.values() if not v)

    return {
        "passes": violations <= 1,  # Se permite 1 violación
        "violations": violations,
        "rules": rules,
        "values": {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba},
    }


def calculate_bbb_score(smiles: str) -> dict:
    """
    Estima la probabilidad de penetración de la Barrera Hematoencefálica (BBB).
    Sistema de 5 criterios basado en Clark 2003.

    Returns:
        Dict con score (0-5), penetra (bool), y detalles por criterio.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"score": 0, "penetra": False, "criteria": {}}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    criteria = {
        "MW ≤ 400": mw <= 400,
        "1 ≤ LogP ≤ 3": 1 <= logp <= 3,
        "TPSA ≤ 90": tpsa <= 90,
        "HBD ≤ 3": hbd <= 3,
        "RotBonds ≤ 8": rot <= 8,
    }

    score = sum(criteria.values())
    return {"score": score, "penetra": score >= 4, "criteria": criteria}


def check_pains(smiles: str) -> dict:
    """
    Verifica si la molécula contiene patrones PAINS (Pan Assay INterference).
    Estos son substructuras que dan falsos positivos en screening.

    Returns:
        Dict con passes (True = libre de PAINS) y alertas.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"passes": False, "alerts": ["Invalid SMILES"]}

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    matches = catalog.GetMatches(mol)
    alerts = [m.GetDescription() for m in matches]

    return {
        "passes": len(alerts) == 0,
        "alerts": alerts,
    }


def full_admet_filter(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    qed_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el pipeline completo de filtros ADMET a un DataFrame de candidatos.

    Filtros:
        1. QED ≥ qed_threshold
        2. Lipinski Rule of 5 (≤ 1 violación)
        3. BBB score ≥ 4/5 (penetración probable)
        4. PAINS-free (sin alertas de interferencia)

    Args:
        df: DataFrame con columna de SMILES.
        smiles_col: Nombre de la columna de SMILES.
        qed_threshold: Umbral mínimo de QED.

    Returns:
        Tupla de (df_passed, df_all_with_scores).
    """
    results = []
    total = len(df)

    logger.info(f"🧪 Evaluando filtros ADMET para {total} candidatos...")

    for i, row in df.iterrows():
        smiles = row[smiles_col]

        qed = calculate_qed(smiles)
        lipinski = check_lipinski(smiles)
        bbb = calculate_bbb_score(smiles)
        pains = check_pains(smiles)

        passes_all = (
            qed >= qed_threshold
            and lipinski["passes"]
            and bbb["penetra"]
            and pains["passes"]
        )

        results.append({
            **row.to_dict(),
            "QED": round(qed, 3),
            "Lipinski_passes": lipinski["passes"],
            "Lipinski_violations": lipinski["violations"],
            "BBB_score": bbb["score"],
            "BBB_penetration": bbb["penetra"],
            "PAINS_free": pains["passes"],
            "PAINS_alerts": "; ".join(pains["alerts"]) if pains["alerts"] else "",
            "ADMET_verdict": "✅ Viable" if passes_all else "❌ Descartado",
            "passes_all_filters": passes_all,
        })

    df_all = pd.DataFrame(results)
    df_passed = df_all[df_all["passes_all_filters"]].copy()

    logger.info(
        f"✅ Filtros ADMET: {len(df_passed)}/{total} candidatos pasaron todos los filtros"
    )

    return df_passed, df_all
