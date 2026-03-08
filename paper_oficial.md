# Predicción Computacional de Inhibidores de Transportadores de Monoaminas para el Tratamiento de la Depresión Mayor: Un Enfoque QSAR Multi-Target con Arquitectura de Agentes Inteligentes

---

**Autor:** Álvaro Vladimir Ocampo Pulido · C.C. 1.107.078.609
**Afiliación:** Colombia
**Fecha:** Marzo 2026
**Palabras clave:** QSAR, XGBoost, antidepresivos, SERT, NAT, DAT, ADMET, machine learning, drug discovery

---

## Resumen

Se presenta un pipeline computacional para la predicción de inhibidores de los tres principales transportadores de monoaminas (SERT, NAT, DAT) implicados en la depresión mayor. El sistema emplea un enfoque QSAR (Quantitative Structure-Activity Relationship) basado en XGBoost, integrando 887 features moleculares — fingerprints ECFP4 filtrados por varianza y 42 descriptores fisicoquímicos normalizados — con optimización bayesiana de hiperparámetros mediante Optuna. Se implementó una arquitectura multi-agente de 5 fases que automatiza el flujo completo desde la adquisición de datos en ChEMBL hasta el filtrado farmacológico ADMET (QED, Lipinski, BBB multi-criterio, PAINS). La metodología replica y extiende el trabajo de la Universidad Icesi incorporando el análisis simultáneo de tres targets y correcciones documentadas en el modelo de penetración de barrera hematoencefálica.

---

## 1. Introducción

### 1.1 Contexto Clínico

La depresión mayor (MDD) afecta a más de 280 millones de personas globalmente (OMS, 2023). Los tratamientos farmacológicos actuales se centran en la modulación de tres neurotransmisores monoaminérgicos:

| Neurotransmisor | Transportador | ChEMBL ID | Clase de Fármaco |
|-----------------|--------------|-----------|------------------|
| Serotonina (5-HT) | SERT | CHEMBL228 | SSRIs (Fluoxetina, Sertralina) |
| Norepinefrina (NE) | NAT/NET | CHEMBL222 | SNRIs (Venlafaxina, Duloxetina) |
| Dopamina (DA) | DAT | CHEMBL238 | NDRIs (Bupropion) |

Aproximadamente el 30% de los pacientes no responden a los SSRIs de primera línea, lo que motiva la exploración computacional de inhibidores de múltiples transportadores.

### 1.2 Objetivo

Desarrollar un sistema computacional automatizado que:
1. Prediga la actividad biológica (pIC₅₀) de compuestos candidatos contra SERT, NAT y DAT
2. Filtre candidatos por propiedades ADMET farmacológicamente relevantes
3. Identifique compuestos con perfiles de selectividad preferencial

---

## 2. Materiales y Métodos

### 2.1 Adquisición de Datos

Los datos bioactivos se obtuvieron de la base de datos ChEMBL (v34) mediante su API REST, con paginación automática de 500 registros/página.

**Criterios de inclusión:**
- Actividades reportadas como IC₅₀ o EC₅₀
- pChEMBL value disponible
- SMILES canónico válido (verificado con RDKit 2024.03)
- Sin comentarios de validez problemáticos ("Outside typical range", "Potential transcription error")

### 2.2 Curación de Datos

El proceso de curación incluyó:

1. **Validación de SMILES:** Canonicalización con RDKit, descarte de estructuras no parseables
2. **Deduplicación:** Para compuestos con mediciones múltiples, se conservó la mediana de pActivity
3. **Eliminación de outliers:** Método IQR adaptativo con cotas biológicas:
   ```
   lower = max(Q1 − 1.5×IQR, 3.0)
   upper = min(Q3 + 1.5×IQR, 12.0)
   ```
   Este enfoque se adapta automáticamente a la distribución de actividad de cada target, evitando rangos fijos que podrían ser inapropiados.

### 2.3 Feature Engineering

Se generó una matriz combinada de ~886 features por molécula:

**Fingerprints ECFP4:**
- Morgan fingerprints (radio=2, 2,048 bits) generados con `rdFingerprintGenerator`
- Filtrado por varianza (umbral=0.01) para eliminar bits constantes → ~844 bits informativos

**Descriptores Fisicoquímicos (42):**

| Categoría | Descriptores | Cantidad |
|-----------|-------------|----------|
| Fisicoquímicos | MW, LogP, TPSA, MR | 4 |
| Topológicos | HBD, HBA, RotBonds, Rings, AromaticRings | 5 |
| Constitucionales | HeavyAtoms, FractionCSP3, Heteroatoms, AliphaticRings | 4 |
| Electrónicos | Gasteiger (max, min, mean), BertzCT | 4 |
| Lipofilicidad | CrippenLogP, CrippenMR | 2 |
| Complejidad | BalabanJ, HallKierAlpha, Kappa1/2/3 | 5 |
| QED | Drug-likeness score | 1 |
| Cargas parciales | Max/Min/MaxAbs/MinAbs | 4 |
| VSA | PEOE_VSA1/2, SMR_VSA1, SlogP_VSA1, EState_VSA1 | 5 |
| Topológicos (Chi) | Chi0, Chi1 | 2 |
| Electrónicos | NumValence, NumRadical | 2 |
| Fragmentos | NH, halogen, ether, benzene, amide | 5 |

- Normalización: `StandardScaler` (µ=0, σ=1)
- Tratamiento de NaN: Imputación por mediana de columna (no cero, para evitar sesgo)

### 2.4 Modelado Predictivo

**Algoritmo:** XGBRegressor (XGBoost)

**Validación:** Split 80/20 estratificado + 5-fold cross-validation

**Métricas:**
- R² (coeficiente de determinación)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Optimización de Hiperparámetros:**
Búsqueda bayesiana con Optuna (50 trials, 5-fold CV):

| Hiperparámetro | Rango |
|---------------|-------|
| n_estimators | 100–1000 |
| max_depth | 3–10 |
| learning_rate | 0.01–0.3 |
| subsample | 0.6–1.0 |
| colsample_bytree | 0.6–1.0 |
| min_child_weight | 1–10 |
| gamma | 0.0–5.0 |
| reg_alpha | 0.0–10.0 |
| reg_lambda | 0.0–10.0 |

### 2.5 Filtrado ADMET

Los candidatos se evaluaron con 4 filtros farmacológicos:

| Filtro | Criterio | Justificación |
|--------|---------|---------------|
| **QED** | ≥ 0.5 | Quantitative Estimate of Drug-likeness (Bickerton et al., 2012) |
| **Lipinski** | ≤ 1 violación | Rule of Five para biodisponibilidad oral |
| **BBB** | Score ≥ 4/5 | Sistema multi-criterio Clark 2003 (MW≤400, LogP 1-3, TPSA≤90, HBD≤3, RotBonds≤8) |
| **PAINS** | Sin alertas | Pan Assay Interference patterns (Baell & Holloway, 2010) |

> **Nota sobre BBB:** Se adoptó deliberadamente un sistema de 5 criterios independientes en lugar de la regresión lineal de Clark (BBB = 0.152×LogP − 0.0148×TPSA + 0.139) debido a que esta última produce falsos negativos documentados en moléculas con actividad neuroactiva confirmada (véase Sección 4.2).

---

## 3. Implementación

### 3.1 Arquitectura Multi-Agente

El sistema se implementó como una arquitectura de 5 agentes especializados en Python puro, cada uno con un ciclo de vida estricto:

```
validate_inputs() → execute() → validate_outputs() → generate_report()
```

| Agente | Fase | Skills Utilizados |
|--------|------|-------------------|
| DataAgent | Adquisición | chembl_scraper, smiles_validator |
| FeatureAgent | Engineering | fingerprint_calculator, descriptor_calculator |
| TrainingAgent | Entrenamiento | model_trainer |
| OptimizationAgent | Optimización | optuna_tuner, model_trainer |
| ADMETAgent | Filtrado | admet_filter |

### 3.2 Multi-Target

El pipeline se ejecuta independientemente para cada target (SERT, NAT, DAT), generando datasets, modelos y reportes separados. Esto permite:
- Comparación de perfiles de actividad entre targets
- Identificación de compuestos multi-target (que inhiban >1 transportador)
- Análisis de selectividad

---

## 4. Resultados

### 4.1 Datasets Obtenidos

*[Los datos específicos se completarán después de la ejecución del pipeline]*

| Target | Registros Crudos | IC₅₀ Curados | EC₅₀ Curados |
|--------|-----------------|-------------|-------------|
| SERT | — | — | — |
| NAT | — | — | — |
| DAT | — | — | — |

### 4.2 Análisis del Modelo BBB

Durante el desarrollo se identificó que la regresión lineal de Clark para BBB:

```
BBB = 0.152 × LogP − 0.0148 × TPSA + 0.139
```

Produce un **falso negativo** para el LSD (C₂₀H₂₅N₃O), una molécula con penetración BBB confirmada experimentalmente:

```
BBB(LSD) = 0.152(2.95) − 0.0148(49.3) + 0.139 = −0.142 → "No penetra"
```

**Causa raíz:** El modelo lineal de 2 variables no captura la contribución de la rigidez molecular (solo 2 enlaces rotables), el tamaño favorable (MW=323), ni la distribución tridimensional de la polaridad.

**Solución implementada:** Sistema multi-criterio de 5 puntos que evalúa cada propiedad independientemente, asignando BBB(LSD) = 5/5 (correcto).

---

## 5. Discusión

### 5.1 Ventajas del Enfoque Multi-Target

A diferencia de pipelines QSAR convencionales enfocados en un solo target, el análisis simultáneo de SERT/NAT/DAT permite identificar:
- **Compuestos selectivos:** Alta actividad en un target, baja en otros
- **Compuestos multi-target:** Actividad en 2-3 transportadores (relevante para depresión resistente)
- **Patrones estructurales:** Subestructuras asociadas a selectividad

### 5.2 Limitaciones

1. Los datos de ChEMBL provienen de ensayos heterogéneos con protocolos variables
2. El modelo QSAR predice actividad in vitro, no eficacia in vivo
3. Los filtros ADMET son aproximaciones — la evaluación definitiva requiere ensayos experimentales
4. La optimización de Optuna puede no encontrar el óptimo global con 50 trials

---

## 6. Conclusiones

Se desarrolló un pipeline QSAR multi-target automatizado para la predicción de inhibidores de transportadores de monoaminas. El sistema integra adquisición de datos, feature engineering, modelado con XGBoost, optimización bayesiana y filtrado ADMET en una arquitectura modular de agentes. Se documentó y corrigió un defecto en el modelo de penetración BBB de Clark 2003, y se implementaron mejoras en el manejo de outliers y datos faltantes.

---

## Referencias

1. Bickerton, G. R., et al. "Quantifying the chemical beauty of drugs." *Nature Chemistry* 4, 90-98 (2012).
2. Clark, D. E. "In silico prediction of blood-brain barrier permeation." *Drug Discovery Today* 8, 927-933 (2003).
3. Baell, J. B. & Holloway, G. A. "New substructure filters for removal of pan assay interference compounds." *J. Med. Chem.* 53, 2719-2740 (2010).
4. Chen, T. & Guestrin, C. "XGBoost: A scalable tree boosting system." *KDD* (2016).
5. Akiba, T., et al. "Optuna: A next-generation hyperparameter optimization framework." *KDD* (2019).
6. Gaulton, A., et al. "The ChEMBL database in 2017." *Nucleic Acids Research* 45, D945-D954 (2017).
7. Universidad Icesi — Paper original de predicción QSAR de antidepresivos con XGBoost.
