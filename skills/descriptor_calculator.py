"""
Skill: Descriptor Calculator
Calcula los 43 descriptores fisicoquímicos/topológicos del paper de la Icesi.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors

logger = logging.getLogger(__name__)


def _safe_calc(func, mol, default=np.nan):
    """Calcula un descriptor de forma segura, retornando default si falla."""
    try:
        return func(mol)
    except Exception:
        return default


def calculate_43_descriptors(smiles: str) -> Optional[dict]:
    """
    Calcula los 43 descriptores del paper de la Icesi para una molécula.

    Categorías:
        - Fisicoquímicos: MW, LogP, TPSA, MR
        - Topológicos: Donors, Acceptors, RotBonds, Rings, AromaticRings
        - Constitucionales: HeavyAtomCount, FractionCSP3, Heteroatoms, AliphaticRings
        - Electrónicos: Gasteiger charges (max, min, mean), BertzCT
        - Lipofilicidad: MolLogP, Crippen contributions
        - Complejidad: BalabanJ, HallKierAlpha, Kappa1/2/3
        - QED: Quantitative Estimate of Drug-likeness
        - Fragmentos: NH, halogen, ether, benzene, amide

    Args:
        smiles: SMILES canónico.

    Returns:
        Diccionario con 43 valores, o None si el SMILES es inválido.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Pre-calcular cargas de Gasteiger
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [
            mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
            for i in range(mol.GetNumAtoms())
        ]
        charges = [c for c in charges if not np.isnan(c) and not np.isinf(c)]
        gasteiger_max = max(charges) if charges else 0.0
        gasteiger_min = min(charges) if charges else 0.0
        gasteiger_mean = np.mean(charges) if charges else 0.0
    except Exception:
        gasteiger_max = gasteiger_min = gasteiger_mean = 0.0

    descriptors = {
        # ── Fisicoquímicos ──
        "MW": _safe_calc(Descriptors.MolWt, mol),
        "LogP": _safe_calc(Descriptors.MolLogP, mol),
        "TPSA": _safe_calc(Descriptors.TPSA, mol),
        "MR": _safe_calc(Descriptors.MolMR, mol),

        # ── Topológicos ──
        "NumHDonors": _safe_calc(Descriptors.NumHDonors, mol),
        "NumHAcceptors": _safe_calc(Descriptors.NumHAcceptors, mol),
        "NumRotatableBonds": _safe_calc(Descriptors.NumRotatableBonds, mol),
        "RingCount": _safe_calc(Descriptors.RingCount, mol),
        "NumAromaticRings": _safe_calc(Descriptors.NumAromaticRings, mol),

        # ── Constitucionales ──
        "HeavyAtomCount": _safe_calc(Descriptors.HeavyAtomCount, mol),
        "FractionCSP3": _safe_calc(Descriptors.FractionCSP3, mol),
        "NumHeteroatoms": _safe_calc(Descriptors.NumHeteroatoms, mol),
        "NumAliphaticRings": _safe_calc(Descriptors.NumAliphaticRings, mol),

        # ── Electrónicos ──
        "GasteigerMax": gasteiger_max,
        "GasteigerMin": gasteiger_min,
        "GasteigerMean": gasteiger_mean,
        "BertzCT": _safe_calc(Descriptors.BertzCT, mol),

        # ── Lipofilicidad ──
        "MolLogP": _safe_calc(Descriptors.MolLogP, mol),
        "CrippenLogP": _safe_calc(lambda m: rdMolDescriptors.CalcCrippenDescriptors(m)[0], mol),
        "CrippenMR": _safe_calc(lambda m: rdMolDescriptors.CalcCrippenDescriptors(m)[1], mol),

        # ── Complejidad ──
        "BalabanJ": _safe_calc(Descriptors.BalabanJ, mol),
        "HallKierAlpha": _safe_calc(Descriptors.HallKierAlpha, mol),
        "Kappa1": _safe_calc(Descriptors.Kappa1, mol),
        "Kappa2": _safe_calc(Descriptors.Kappa2, mol),
        "Kappa3": _safe_calc(Descriptors.Kappa3, mol),

        # ── QED ──
        "QED": _safe_calc(Descriptors.qed, mol),

        # ── Otros descriptores importantes ──
        "NumValenceElectrons": _safe_calc(Descriptors.NumValenceElectrons, mol),
        "NumRadicalElectrons": _safe_calc(Descriptors.NumRadicalElectrons, mol),
        "MaxPartialCharge": _safe_calc(Descriptors.MaxPartialCharge, mol),
        "MinPartialCharge": _safe_calc(Descriptors.MinPartialCharge, mol),
        "MaxAbsPartialCharge": _safe_calc(Descriptors.MaxAbsPartialCharge, mol),
        "MinAbsPartialCharge": _safe_calc(Descriptors.MinAbsPartialCharge, mol),
        "PEOE_VSA1": _safe_calc(Descriptors.PEOE_VSA1, mol),
        "PEOE_VSA2": _safe_calc(Descriptors.PEOE_VSA2, mol),
        "SMR_VSA1": _safe_calc(Descriptors.SMR_VSA1, mol),
        "SlogP_VSA1": _safe_calc(Descriptors.SlogP_VSA1, mol),
        "EState_VSA1": _safe_calc(Descriptors.EState_VSA1, mol),
        "Chi0": _safe_calc(Descriptors.Chi0, mol),
        "Chi1": _safe_calc(Descriptors.Chi1, mol),

        # ── Fragmentos ──
        "fr_NH": _safe_calc(Fragments.fr_NH0, mol),
        "fr_halogen": _safe_calc(Fragments.fr_halogen, mol),
        "fr_ether": _safe_calc(Fragments.fr_ether, mol),
        "fr_benzene": _safe_calc(Fragments.fr_benzene, mol),
        "fr_amide": _safe_calc(Fragments.fr_amide, mol),
    }

    return descriptors


def calculate_descriptors_batch(
    smiles_list: list,
) -> pd.DataFrame:
    """
    Calcula los 43 descriptores para un lote de moléculas.

    Args:
        smiles_list: Lista de SMILES canónicos.

    Returns:
        DataFrame con 43 columnas de descriptores.
    """
    results = []
    failed = 0

    for i, smiles in enumerate(smiles_list):
        desc = calculate_43_descriptors(smiles)
        if desc is None:
            results.append({})
            failed += 1
        else:
            results.append(desc)

        if (i + 1) % 500 == 0:
            logger.info(f"  ⚗️  {i + 1}/{len(smiles_list)} moléculas procesadas...")

    df = pd.DataFrame(results)

    # Rellenar NaN con 0 para moléculas que fallaron
    df = df.fillna(0)

    logger.info(
        f"📊 Descriptores calculados: {len(df)} moléculas × {len(df.columns)} descriptores "
        f"({failed} fallaron)"
    )

    return df
