"""
Configuración central del Pipeline QSAR Multi-Agente.
Todos los parámetros del pipeline se definen aquí para fácil ajuste.

Soporta múltiples targets: SERT, NAT, DAT.
"""

from pathlib import Path

# ─── Rutas del proyecto ───
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELOS_DIR = BASE_DIR / "modelos"
REPORTES_DIR = BASE_DIR / "reportes"

# Crear directorios si no existen
for d in [DATASETS_DIR, MODELOS_DIR, REPORTES_DIR]:
    d.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════
#  TARGETS — Los 3 transportadores de monoaminas
# ═══════════════════════════════════════════════
TARGETS = {
    "SERT": {
        "chembl_id": "CHEMBL228",
        "name": "Transportador de Serotonina",
        "abbreviation": "SERT",
        "neurotransmitter": "Serotonina (5-HT)",
        "relevance": "Target principal de SSRIs (Fluoxetina, Sertralina, etc.)",
    },
    "NAT": {
        "chembl_id": "CHEMBL222",
        "name": "Transportador de Norepinefrina",
        "abbreviation": "NAT/NET",
        "neurotransmitter": "Norepinefrina (NE)",
        "relevance": "Target de SNRIs (Venlafaxina, Duloxetina) y tricíclicos",
    },
    "DAT": {
        "chembl_id": "CHEMBL238",
        "name": "Transportador de Dopamina",
        "abbreviation": "DAT",
        "neurotransmitter": "Dopamina (DA)",
        "relevance": "Target de NDRIs (Bupropion) y relevante en depresión anhedónica",
    },
}

# Target activo por defecto (para ejecución de un solo target)
DEFAULT_TARGET = "SERT"


# ─── ChEMBL API ───
LIMIT_PER_PAGE = 500
API_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


# ─── Curación de datos ───
MIN_DATASET_SIZE = 500              # Mínimo para cualquier target (SERT tendrá más que NAT/DAT)
PACTIVITY_COLUMN = "pActivity"
SMILES_COLUMN = "canonical_smiles"


# ─── Fingerprints ───
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
VARIANCE_THRESHOLD = 0.01


# ─── Entrenamiento ───
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


# ─── XGBoost defaults ───
XGBOOST_DEFAULTS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ─── Optuna ───
OPTUNA_TRIALS = 50
OPTUNA_SEARCH_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
}


# ─── ADMET / Filtros ───
QED_THRESHOLD = 0.5
LIPINSKI_RULES = {"MW_max": 500, "LogP_max": 5, "HBD_max": 5, "HBA_max": 10}
BBB_THRESHOLD = 0.0


# ═══════════════════════════════════════════════
#  FUNCIONES HELPER — Rutas dinámicas por target
# ═══════════════════════════════════════════════

def get_target_paths(target_key: str) -> dict:
    """
    Genera todas las rutas de archivos para un target dado.

    Args:
        target_key: "SERT", "NAT", o "DAT"

    Returns:
        Dict con todas las rutas de archivos para ese target.
    """
    key = target_key.lower()
    return {
        "dataset_crudo": DATASETS_DIR / f"dataset_{key}_crudo.csv",
        "dataset_ic50": DATASETS_DIR / f"dataset_{key}_ic50_limpio.csv",
        "dataset_ec50": DATASETS_DIR / f"dataset_{key}_ec50_limpio.csv",
        "features_matrix": DATASETS_DIR / f"features_{key}.csv",
        "modelo_ic50": MODELOS_DIR / f"modelo_{key}_ic50.pkl",
        "modelo_ec50": MODELOS_DIR / f"modelo_{key}_ec50.pkl",
        "modelo_ic50_optuna": MODELOS_DIR / f"modelo_{key}_ic50_optuna.pkl",
        "modelo_ec50_optuna": MODELOS_DIR / f"modelo_{key}_ec50_optuna.pkl",
        "reporte_curacion": REPORTES_DIR / f"reporte_curacion_{key}.md",
        "reporte_features": REPORTES_DIR / f"reporte_features_{key}.md",
        "reporte_entrenamiento": REPORTES_DIR / f"reporte_entrenamiento_{key}.md",
        "reporte_optimizacion": REPORTES_DIR / f"reporte_optimizacion_{key}.md",
        "reporte_admet": REPORTES_DIR / f"reporte_admet_{key}.md",
        "candidatos_evaluados": REPORTES_DIR / f"candidatos_evaluados_{key}.csv",
        "candidatos_viables": REPORTES_DIR / f"candidatos_viables_{key}.csv",
    }


# ─── Retrocompatibilidad: Rutas del target por defecto (SERT) ───
_default_paths = get_target_paths(DEFAULT_TARGET)
DATASET_CRUDO = _default_paths["dataset_crudo"]
DATASET_IC50 = _default_paths["dataset_ic50"]
DATASET_EC50 = _default_paths["dataset_ec50"]
FEATURES_MATRIX = _default_paths["features_matrix"]
MODELO_IC50 = _default_paths["modelo_ic50"]
MODELO_EC50 = _default_paths["modelo_ec50"]
MODELO_IC50_OPTUNA = _default_paths["modelo_ic50_optuna"]
MODELO_EC50_OPTUNA = _default_paths["modelo_ec50_optuna"]

# Legado — mantener TARGET_ID y TARGET_NAME para agentes individuales
TARGET_ID = TARGETS[DEFAULT_TARGET]["chembl_id"]
TARGET_NAME = DEFAULT_TARGET
