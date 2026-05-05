# 🧬 Pipeline QSAR Multi-Agente · Antidepresivos

> Predicción computacional de inhibidores del Transportador de Serotonina (SERT) usando Machine Learning — Replicando la metodología del paper de la Universidad Icesi.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Las 5 Fases](#-las-5-fases)
- [Skills Reutilizables](#-skills-reutilizables)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Configuración](#-configuración)
- [Troubleshooting](#-troubleshooting)

---

## 🔬 Descripción

Este proyecto implementa un pipeline **QSAR (Quantitative Structure-Activity Relationship)** para predecir la actividad biológica de compuestos candidatos a antidepresivos contra el transportador de serotonina (STER, CHEMBL228).

El sistema está construido como una arquitectura **multi-agente con skills reutilizables**:

- **5 Agentes** especializados, uno por fase del pipeline
- **8 Skills** modulares que los agentes reutilizan
- **1 Orquestador** (`main.py`) que ejecuta todo en secuencia con validaciones

Cada agente sigue un ciclo de vida estricto:
```
validate_inputs() → execute() → validate_outputs() → generate_report()
```

---

## 🏗️ Arquitectura

```
                    ┌─────────────┐
                    │  main.py    │  ← Orquestador CLI
                    │ (Pipeline)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ config  │       │ agents/ │       │ skills/ │
   │  .py    │       │         │       │         │
   └─────────┘       └────┬────┘       └────┬────┘
                          │                  │
              ┌───────────┴─────┐    ┌──────┴──────────┐
              │ BaseAgent (ABC) │    │ 8 módulos Python │
              │                 │    │ importables      │
              │ ► DataAgent     │◄───│ ► chembl_scraper │
              │ ► FeatureAgent  │    │ ► smiles_valid.  │
              │ ► TrainingAgent │    │ ► fp_calculator  │
              │ ► OptimAgent    │    │ ► desc_calc.     │
              │ ► ADMETAgent    │    │ ► model_trainer  │
              └─────────────────┘    │ ► optuna_tuner   │
                                     │ ► admet_filter   │
                                     │ ► report_gen.    │
                                     └─────────────────┘
```

**Flujo de datos:**
```
DataAgent → datasets/*.csv → FeatureAgent → features_887.csv → TrainingAgent
    → modelos/*.pkl → OptimizationAgent → modelos/*_optuna.pkl → ADMETAgent
    → candidatos_viables.csv → App Streamlit
```

---

## 📦 Requisitos

- **Python** 3.10+
- **RAM** 8 GB mínimo (recomendado 16+ GB para Optuna)
- **SO** Windows / Linux / macOS

### Dependencias Python

| Paquete | Uso |
|---------|-----|
| `rdkit` | Procesamiento molecular (SMILES, fingerprints, descriptores) |
| `pandas` | Manipulación de datos |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Preprocesamiento y métricas |
| `xgboost` | Modelo de Machine Learning |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `requests` | API de ChEMBL |
| `joblib` | Serialización de modelos |
| `scipy` | Estadísticas (correlación) |
| `streamlit` | Interfaz web (app_qsar.py) |
| `matplotlib` / `seaborn` | Visualizaciones |

---

## 🚀 Instalación

### 1. Clonar / Posicionarse en el proyecto

```bash
cd e:\proyecto-ia\antidepresivos
```

### 2. Instalar dependencias

```bash
pip install rdkit pandas numpy scikit-learn xgboost optuna streamlit requests joblib matplotlib seaborn scipy
```

> **Nota sobre RDKit:** Si `pip install rdkit` falla, usa conda:
> ```bash
> conda install -c conda-forge rdkit
> ```

### 3. Verificar instalación

```bash
python -c "from agents.base_agent import BaseAgent; from skills import chembl_scraper; import config; print('OK - Todo instalado correctamente')"
```

---

## 🎮 Uso

### Ejecutar el pipeline completo (Fases 1-5)

```bash
python main.py
```

Esto ejecuta **todo** en secuencia: descarga datos → calcula features → entrena modelos → optimiza → filtra candidatos.

### Ejecutar una sola fase

```bash
python main.py --fase 1    # Solo descarga y curación de datos
python main.py --fase 2    # Solo feature engineering
python main.py --fase 3    # Solo entrenamiento de modelos
python main.py --fase 4    # Solo optimización con Optuna
python main.py --fase 5    # Solo filtros ADMET
```

### Continuar desde una fase (si algo falló)

```bash
python main.py --desde 3   # Ejecuta fases 3, 4 y 5
```

> 💡 Si la Fase 1 ya descargó los datos, no necesitas volver a ejecutarla. Usa `--desde 2` para continuar.

### Ejecutar un agente individualmente

```bash
python -m agents.data_agent           # Solo DataAgent
python -m agents.feature_agent        # Solo FeatureAgent
python -m agents.training_agent       # Solo TrainingAgent
python -m agents.optimization_agent   # Solo OptimizationAgent
python -m agents.admet_agent          # Solo ADMETAgent
```

### Lanzar la app Streamlit

```bash
streamlit run app_qsar.py
```

---

## 📊 Las 5 Fases

### Fase 1 — Adquisición de Datos (`DataAgent`)

| Aspecto | Detalle |
|---------|---------|
| **Skills usados** | `chembl_scraper`, `smiles_validator`, `report_generator` |
| **Input** | API de ChEMBL (CHEMBL228 — SERT) |
| **Output** | `datasets/dataset_sert_crudo.csv`, `datasets/dataset_ic50_limpio.csv`, `datasets/dataset_ec50_limpio.csv` |
| **Meta** | ≥ 2,000 registros limpios de IC₅₀ |

**Qué hace:**
1. Scrapea **todas** las actividades biológicas del SERT con paginación automática
2. Valida cada SMILES con RDKit
3. Elimina duplicados (mediana de pActivity)
4. Filtra outliers (pActivity entre 3 y 12)
5. Separa en datasets IC₅₀ y EC₅₀

---

### Fase 2 — Feature Engineering (`FeatureAgent`)

| Aspecto | Detalle |
|---------|---------|
| **Skills usados** | `fingerprint_calculator`, `descriptor_calculator`, `report_generator` |
| **Input** | `datasets/dataset_ic50_limpio.csv` |
| **Output** | `datasets/features_887.csv` |
| **Meta** | Matriz de ~887 features por molécula |

**Qué hace:**
1. Genera fingerprints ECFP4 (2,048 bits)
2. Filtra por varianza mínima (2,048 → ~844 bits)
3. Calcula 43 descriptores fisicoquímicos (MW, LogP, TPSA, QED, etc.)
4. Normaliza descriptores con `StandardScaler`
5. Concatena: 844 bits + 43 descriptores = ~887 features

---

### Fase 3 — Entrenamiento Dual (`TrainingAgent`)

| Aspecto | Detalle |
|---------|---------|
| **Skills usados** | `model_trainer`, `report_generator` |
| **Input** | `datasets/features_887.csv` |
| **Output** | `modelos/modelo_ic50.pkl`, `modelos/modelo_ec50.pkl` |
| **Meta** | R² > 0.5 en ambos modelos |

**Qué hace:**
1. Entrena XGBRegressor para IC₅₀ (eficacia bioquímica)
2. Entrena XGBRegressor para EC₅₀ (eficacia celular) si hay datos suficientes
3. Evalúa con R², RMSE, MAE y 5-fold cross-validation

---

### Fase 4 — Optimización (`OptimizationAgent`)

| Aspecto | Detalle |
|---------|---------|
| **Skills usados** | `optuna_tuner`, `model_trainer`, `report_generator` |
| **Input** | `datasets/features_887.csv`, modelos base |
| **Output** | `modelos/modelo_ic50_optuna.pkl` |
| **Meta** | Mejorar R² vs modelo default |

**9 hiperparámetros optimizados:**
`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`

---

### Fase 5 — Filtro ADMET (`ADMETAgent`)

| Aspecto | Detalle |
|---------|---------|
| **Skills usados** | `admet_filter`, `report_generator` |
| **Input** | `datasets/dataset_ic50_limpio.csv` |
| **Output** | `reportes/candidatos_viables.csv` |
| **Meta** | Lista final de candidatos farmacológicamente viables |

**4 filtros aplicados:**

| Filtro | Criterio |
|--------|----------|
| **QED** | ≥ 0.5 (Drug-likeness) |
| **Lipinski** | ≤ 1 violación (Rule of 5) |
| **BBB** | Score > 0 (penetración barrera hematoencefálica) |
| **PAINS** | Sin alertas de interferencia |

---

## 🔧 Skills Reutilizables

Cada skill es un módulo Python independiente que puede usarse fuera del pipeline:

```python
# Ejemplo: usar el scraper por separado
from skills.chembl_scraper import scrape_target
df = scrape_target("CHEMBL228")

# Ejemplo: validar un SMILES
from skills.smiles_validator import validate_smiles
canonical = validate_smiles("c1ccccc1")  # → "c1ccccc1"

# Ejemplo: calcular descriptores de una molécula
from skills.descriptor_calculator import calculate_43_descriptors
desc = calculate_43_descriptors("CC(=O)Oc1ccccc1C(=O)O")  # Aspirina
print(desc["MW"], desc["LogP"], desc["QED"])

# Ejemplo: verificar filtros ADMET
from skills.admet_filter import check_lipinski, calculate_qed
print(check_lipinski("CC(=O)Oc1ccccc1C(=O)O"))
print(calculate_qed("CC(=O)Oc1ccccc1C(=O)O"))
```

---

## 📁 Estructura del Proyecto

```
e:\proyecto-ia\antidepresivos\
│
├── main.py                        # Orquestador principal (CLI)
├── config.py                      # Configuración centralizada
├── README.md                      # Este archivo
│
├── agents/                        # Agentes (uno por fase)
│   ├── __init__.py
│   ├── base_agent.py              # Clase abstracta BaseAgent
│   ├── data_agent.py              # Fase 1: Datos
│   ├── feature_agent.py           # Fase 2: Features
│   ├── training_agent.py          # Fase 3: Entrenamiento
│   ├── optimization_agent.py      # Fase 4: Optuna
│   └── admet_agent.py             # Fase 5: ADMET
│
├── skills/                        # Capacidades reutilizables
│   ├── __init__.py
│   ├── chembl_scraper.py          # Scraping ChEMBL con paginación
│   ├── smiles_validator.py        # Validación y curación SMILES
│   ├── fingerprint_calculator.py  # ECFP4 + filtro varianza
│   ├── descriptor_calculator.py   # 43 descriptores Icesi
│   ├── model_trainer.py           # XGBoost + métricas + CV
│   ├── optuna_tuner.py            # Bayesian tuning
│   ├── admet_filter.py            # QED + Lipinski + BBB + PAINS
│   └── report_generator.py        # Reportes markdown
│
├── datasets/                      # Datos (generado automáticamente)
│   ├── dataset_sert_crudo.csv     # Datos crudos de ChEMBL
│   ├── dataset_ic50_limpio.csv    # IC₅₀ curado
│   ├── dataset_ec50_limpio.csv    # EC₅₀ curado
│   └── features_887.csv           # Matriz de features
│
├── modelos/                       # Modelos serializados
│   ├── modelo_ic50.pkl            # Modelo base IC₅₀
│   ├── modelo_ec50.pkl            # Modelo base EC₅₀
│   ├── modelo_ic50_optuna.pkl     # Modelo optimizado IC₅₀
│   └── modelo_ec50_optuna.pkl     # Modelo optimizado EC₅₀
│
├── reportes/                      # Reportes generados
│   ├── reporte_curacion.md
│   ├── reporte_features.md
│   ├── reporte_entrenamiento.md
│   ├── reporte_optimizacion.md
│   ├── reporte_admet.md
│   ├── candidatos_evaluados.csv
│   └── candidatos_viables.csv
│
└── app_qsar.py                    # Interfaz Streamlit
```

---

## ⚙️ Configuración

Todos los parámetros están centralizados en `config.py`:

```python
# Target
TARGET_ID = "CHEMBL228"        # Transportador de Serotonina (SERT)

# Datos
MIN_DATASET_SIZE = 2000        # Mínimo de registros (paper: 2,034)

# Features
FINGERPRINT_BITS = 2048        # ECFP4 estándar
VARIANCE_THRESHOLD = 0.01     # Filtro de bits constantes

# Entrenamiento
TEST_SIZE = 0.2                # 80/20 split
CV_FOLDS = 5                   # 5-fold cross-validation

# Optuna
OPTUNA_TRIALS = 50             # Trials bayesianos

# ADMET
QED_THRESHOLD = 0.5            # Drug-likeness mínimo
```

Para cambiar el target (ej: 5-HT1A receptor), solo edita `TARGET_ID` en `config.py`.

---

## 🛠️ Troubleshooting

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: rdkit` | Instalar con conda: `conda install -c conda-forge rdkit` |
| `ModuleNotFoundError: optuna` | `pip install optuna` |
| `UnicodeEncodeError` en consola | Ejecutar con: `set PYTHONIOENCODING=utf-8` antes del comando |
| Fase 1 tarda mucho | Normal — descarga ~4,000+ registros de ChEMBL con rate limiting |
| R² negativo en Fase 3 | El dataset puede ser muy pequeño. Verifica que Fase 1 produjo ≥ 2,000 registros |
| Optuna falla por memoria | Reducir `OPTUNA_TRIALS` en `config.py` (ej: 20 en vez de 50) |
| Ningún candidato pasa ADMET | Relajar `QED_THRESHOLD` en `config.py` (ej: 0.3 en vez de 0.5) |

---

## 📚 Referencias

- **Paper base:** Universidad Icesi — Predicción QSAR de antidepresivos con XGBoost
- **ChEMBL:** [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
- **RDKit:** [https://www.rdkit.org/](https://www.rdkit.org/)
- **XGBoost:** [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **Optuna:** [https://optuna.org/](https://optuna.org/)

---

*Pipeline construido con arquitectura multi-agente en Python puro.*
