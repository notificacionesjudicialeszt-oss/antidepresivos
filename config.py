"""
Configuración central del Pipeline QSAR Multi-Agente.
Todos los parámetros del pipeline se definen aquí para fácil ajuste.
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


# ─── ChEMBL / Target ───
TARGET_ID = "CHEMBL228"            # SERT — Transportador de Serotonina
TARGET_NAME = "SERT"
LIMIT_PER_PAGE = 500               # Registros por página en la API
API_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


# ─── Curación de datos ───
MIN_DATASET_SIZE = 2000             # Mínimo de registros limpios (paper: 2,034)
PACTIVITY_COLUMN = "pActivity"
SMILES_COLUMN = "canonical_smiles"


# ─── Fingerprints ───
FINGERPRINT_BITS = 2048             # ECFP4 estándar
FINGERPRINT_RADIUS = 2             # Radio de Morgan (ECFP4 = radio 2)
VARIANCE_THRESHOLD = 0.01          # Umbral para filtrar bits constantes


# ─── Entrenamiento ───
TEST_SIZE = 0.2                    # 80% train / 20% test
RANDOM_STATE = 42
CV_FOLDS = 5                      # 5-fold cross-validation


# ─── XGBoost defaults ───
XGBOOST_DEFAULTS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ─── Optuna ───
OPTUNA_TRIALS = 50                 # Número de trials (paper: 50)
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
QED_THRESHOLD = 0.5                # Paper: QED ≥ 0.5

LIPINSKI_RULES = {
    "MW_max": 500,
    "LogP_max": 5,
    "HBD_max": 5,
    "HBA_max": 10,
}

# Clark 2003 BBB score
BBB_THRESHOLD = 0.0                # Score > 0 = probable penetración BBB


# ─── Archivos de salida ───
DATASET_CRUDO = DATASETS_DIR / "dataset_sert_crudo.csv"
DATASET_IC50 = DATASETS_DIR / "dataset_ic50_limpio.csv"
DATASET_EC50 = DATASETS_DIR / "dataset_ec50_limpio.csv"
FEATURES_MATRIX = DATASETS_DIR / "features_887.csv"

MODELO_IC50 = MODELOS_DIR / "modelo_ic50.pkl"
MODELO_EC50 = MODELOS_DIR / "modelo_ec50.pkl"
MODELO_IC50_OPTUNA = MODELOS_DIR / "modelo_ic50_optuna.pkl"
MODELO_EC50_OPTUNA = MODELOS_DIR / "modelo_ec50_optuna.pkl"
