# 🔒 Paper Privado — Documentación Técnica Interna
## Pipeline QSAR Multi-Target | Antidepresivos

> Este documento contiene toda la información técnica interna: decisiones de diseño, bugs encontrados y corregidos, limitaciones conocidas, y notas para futuro desarrollo. **No compartir externamente.**

---

## 1. Decisiones de Arquitectura

### ¿Por qué Python puro en vez de CrewAI/LangGraph?

Se evaluaron 3 opciones:

| Opción | Ventaja | Desventaja | Decisión |
|--------|---------|------------|----------|
| Python puro (ABC + clases) | Control total, sin dependencias extra, determinístico | Más código manual | ✅ **Elegido** |
| CrewAI | Framework listo, roles de agentes | Requiere API key LLM, overhead, latencia | ❌ |
| LangGraph | Grafos de estado, visualización | Complejidad innecesaria para pipeline lineal | ❌ |

**Razón principal:** El pipeline es **determinístico** — no necesita un LLM para decidir qué hacer. Cada fase tiene inputs y outputs fijos. Un framework de agentes LLM añadiría costo ($), latencia, y puntos de falla sin beneficio real.

### Patrón de diseño: Agent-Skill

```
BaseAgent (ABC)
├── validate_inputs()    ← ¿Tengo lo que necesito?
├── execute()            ← Skill calls aquí
├── validate_outputs()   ← ¿El resultado tiene sentido?
└── generate_report()    ← Documentar para humanos
```

Los **Skills** son módulos stateless que pueden usarse fuera del pipeline. Los **Agents** orquestan skills y manejan errores.

---

## 2. Bugs Encontrados y Corregidos

### 🔴 BUG-001: BBB False Negative (CRÍTICO)

**Fecha descubierta:** 2026-03-07
**Archivos afectados:** `panel.py`, `admet_filter.py`
**Descubierto por:** Test manual con LSD en el panel Streamlit

**Descripción:**
La fórmula lineal de Clark 2003 (`0.152*LogP - 0.0148*TPSA + 0.139`) produce falsos negativos para moléculas neuroactivas conocidas.

**Caso de prueba:**
```
LSD (C₂₀H₂₅N₃O):
  LogP = 2.95, TPSA = 49.3
  BBB = 0.152(2.95) - 0.0148(49.3) + 0.139 = -0.142
  Resultado: "No penetra" ← INCORRECTO
  Realidad: LSD cruza BBB extremadamente bien
```

**Análisis de causa raíz:**
- La fórmula usa solo 2 variables (LogP, TPSA)
- El coeficiente de TPSA (-0.0148) domina al de LogP (0.152)
- No considera: rigidez molecular, tamaño, distribución 3D de cargas
- Para LSD, la alta rigidez (2 rotbonds) y tamaño pequeño (MW=323) favorecen BBB, pero la fórmula no lo captura

**Fix aplicado:**
Sistema de 5 criterios independientes (cada uno sí/no):
```python
MW ≤ 400, 1≤LogP≤3, TPSA≤90, HBD≤3, RotBonds≤8
Score ≥ 4 → penetra
```
LSD: 5/5 ✅

**Impacto:** Candidatos válidos estaban siendo descartados en Fase 5. La corrección puede aumentar significativamente el número de candidatos viables.

---

### 🟡 BUG-002: NaN → 0 en Descriptores (MEDIO)

**Fecha:** 2026-03-07
**Archivo:** `descriptor_calculator.py` línea 159

**Antes:**
```python
df = df.fillna(0)
```

**Problema:**
- Un descriptor que falla no tiene valor 0; tiene valor desconocido
- Rellenar con 0 distorsiona la media y std del StandardScaler
- Ejemplo: Si BalabanJ falla para una molécula (RDKit error), ponerle 0 la hace parecer "simple" cuando en realidad es desconocida

**Fix:** Imputación por mediana de columna:
```python
for col in df.columns:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
```

---

### 🟡 BUG-003: Rango de Outliers Fijo (MEDIO)

**Archivo:** `smiles_validator.py` línea 84-86

**Antes:**
```python
df = df[(df["pActivity"] >= 3.0) & (df["pActivity"] <= 12.0)]
```

**Problema:**
- DAT puede tener distribuciones de pActivity diferentes a SERT
- Un compuesto con pActivity=2.9 (interesante) se descarta silenciosamente

**Fix:** IQR adaptativo con cotas biológicas:
```python
Q1, Q3 = df["pActivity"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower = max(Q1 - 1.5*IQR, 3.0)
upper = min(Q3 + 1.5*IQR, 12.0)
```

---

### 🔵 BUG-004: LogP Duplicado (BAJO)

**Archivo:** `descriptor_calculator.py`

**Antes:** `LogP` (Wildman-Crippen) y `MolLogP` (idéntico) aparecían como 2 features separadas.

**Fix:** Eliminado `MolLogP`, conservado solo `LogP` + `CrippenLogP` (que usa algoritmo diferente).

---

### 🔵 BUG-005: Deprecation Warning MorganFingerprint (BAJO)

**Archivo:** `fingerprint_calculator.py`

**Antes:** `AllChem.GetMorganFingerprintAsBitVect()` — deprecado en RDKit 2024+

**Fix:** Migrado a `rdFingerprintGenerator.GetMorganGenerator().GetFingerprintAsNumPy()`

---

## 3. Limitaciones Conocidas (No Corregidas)

### L-001: Filtro de varianza sin considerar target

`VarianceThreshold(0.01)` elimina bits del fingerprint basándose solo en varianza global. Un bit con varianza baja podría tener alta correlación con pActivity pero ser descartado.

**Alternativa futura:** `SelectKBest(mutual_info_regression, k=800)`

### L-002: Sin manejo de desbalanceo de actividad

Si hay muchas moléculas inactivas (pActivity < 5) y pocas activas (> 7), el modelo aprende mejor los inactivos. No se aplica sobremuestreo ni pesos.

**Alternativa futura:** `sample_weight` proporcional a pActivity o SMOTE para regresión.

### L-003: Datos heterogéneos de ChEMBL

Los datos provienen de cientos de papers con diferentes protocolos experimentales. La variabilidad inter-ensayo añade ruido que el modelo no puede distinguir de la señal.

### L-004: Un solo algoritmo (XGBoost)

No se compararon alternativas (Random Forest, LightGBM, redes neuronales). Un ensemble podría mejorar la predicción.

---

## 4. Hitos del Desarrollo

| Fecha | Hito |
|-------|------|
| 2026-03-07 18:00 | Arquitectura multi-agente definida |
| 2026-03-07 18:15 | 8 skills + 5 agents implementados |
| 2026-03-07 18:20 | Orquestador + CLI listo |
| 2026-03-07 18:33 | Panel Streamlit creado |
| 2026-03-07 18:46 | Git init + primer commit |
| 2026-03-07 18:58 | Multi-target (SERT/NAT/DAT) añadido |
| 2026-03-07 19:41 | Bug BBB descubierto y corregido |
| 2026-03-07 20:16 | Auditoría completa + correcciones finales |

---

## 5. Estructura Final de Archivos

```
antidepresivos/
├── config.py                   # 3 targets, paths dinámicas, get_target_paths()
├── main.py                     # --target SERT/NAT/DAT/[todos] --fase --desde
├── panel.py                    # Dashboard Streamlit multi-target
├── app_qsar.py                 # App original de análisis molecular
│
├── agents/
│   ├── base_agent.py           # ABC: validate → execute → validate → report
│   ├── data_agent.py           # Fase 1: Scraping + Curación
│   ├── feature_agent.py        # Fase 2: ECFP4 + 42 descriptores
│   ├── training_agent.py       # Fase 3: XGBoost dual (IC50/EC50)
│   ├── optimization_agent.py   # Fase 4: Optuna bayesiano
│   └── admet_agent.py          # Fase 5: QED + Lipinski + BBB + PAINS
│
├── skills/
│   ├── chembl_scraper.py       # Paginación + retry + rate limit
│   ├── smiles_validator.py     # Curación + IQR outliers
│   ├── fingerprint_calculator.py  # MorganGenerator (API moderna)
│   ├── descriptor_calculator.py   # 42 desc + NaN→mediana
│   ├── model_trainer.py        # XGBoost + CV + serialización
│   ├── optuna_tuner.py         # 50 trials bayesianos
│   ├── admet_filter.py         # BBB 5-criterios (corregido)
│   └── report_generator.py     # Markdown reports
│
├── datasets/                   # dataset_{target}_{tipo}.csv
├── modelos/                    # modelo_{target}_{tipo}.pkl
├── reportes/                   # reporte_{fase}_{target}.md
│
├── paper_oficial.md
├── paper_privado.md            # ← Este documento
└── README.md
```

---

## 6. Próximos Pasos Sugeridos

1. **Ejecutar pipeline completo** para los 3 targets y llenar tablas de resultados en el paper oficial
2. **Análisis de selectividad:** Comparar predicciones entre targets para identificar compuestos multi-target
3. **Ensemble de modelos:** Añadir Random Forest y LightGBM para comparación
4. **Feature importance:** Identificar qué descriptores son más predictivos por target
5. **Validación externa:** Testear predicciones contra compuestos con actividad conocida no incluidos en training
6. **Actualizar app_qsar.py** para consumir los modelos multi-target

---

*Documento interno — Pipeline QSAR Multi-Agente v2.0*
*© 2026 Álvaro Vladimir Ocampo Pulido · C.C. 1.107.078.609 · Todos los derechos reservados*
