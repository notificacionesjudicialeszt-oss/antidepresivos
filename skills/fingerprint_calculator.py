"""
Skill: Fingerprint Calculator
Genera ECFP4 (Morgan fingerprints) y aplica filtrado por varianza.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


def calculate_fingerprints(
    smiles_list: list,
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """
    Genera matriz de Morgan fingerprints (ECFP4) para una lista de SMILES.

    Args:
        smiles_list: Lista de SMILES canónicos.
        n_bits: Número de bits del fingerprint.
        radius: Radio de Morgan (2 = ECFP4).

    Returns:
        Matriz numpy de forma (n_molecules, n_bits).
    """
    # Usar la API moderna de RDKit (MorganGenerator)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )

    fps = []
    failed = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fps.append(np.zeros(n_bits))
            failed += 1
        else:
            fp = generator.GetFingerprintAsNumPy(mol)
            fps.append(fp)

    if failed > 0:
        logger.warning(f"⚠️  {failed} moléculas no pudieron ser procesadas (vector cero)")

    matrix = np.array(fps)
    logger.info(f"🧬 Fingerprints generados: {matrix.shape[0]} moléculas × {matrix.shape[1]} bits")
    return matrix


def filter_by_variance(
    fp_matrix: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra bits de fingerprint por varianza mínima.

    Args:
        fp_matrix: Matriz de fingerprints (n, bits).
        threshold: Umbral de varianza mínima.

    Returns:
        Tuple de (matriz_filtrada, indices_de_bits_retenidos).
    """
    selector = VarianceThreshold(threshold=threshold)
    filtered = selector.fit_transform(fp_matrix)
    retained_indices = selector.get_support(indices=True)

    logger.info(
        f"🔬 Filtro de varianza (umbral={threshold}): "
        f"{fp_matrix.shape[1]} bits → {filtered.shape[1]} bits informativos"
    )

    return filtered, retained_indices
