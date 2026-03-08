# 🧬 NeuroPharma AI — Elevator Pitch

---

## El Problema (10 segundos)

> **280 millones** de personas sufren depresión en el mundo. El **30% no responde** al tratamiento actual. Desarrollar un nuevo antidepresivo cuesta **$2.6 billones USD** y toma **12 años** en promedio.

---

## La Solución (20 segundos)

Construimos un **motor de IA** que analiza miles de moléculas en minutos — no en años — para identificar cuáles tienen la mayor probabilidad de funcionar como antidepresivos, **antes de pisar un laboratorio**.

Nuestro sistema evalúa simultáneamente los **3 targets neuroquímicos** de la depresión:

| 🔵 Serotonina (SERT) | 🟠 Norepinefrina (NAT) | 🟢 Dopamina (DAT) |
|:---:|:---:|:---:|
| SSRIs como Prozac | SNRIs como Cymbalta | NDRIs como Wellbutrin |

---

## ¿Cómo Funciona? (15 segundos)

```
Datos de 4,000+ moléculas de ChEMBL (base mundial)
         ↓
887 características moleculares por compuesto
         ↓
IA predictiva (XGBoost + Optuna) → Precisión del modelo
         ↓
4 filtros farmacológicos reales (QED + Lipinski + BBB + PAINS)
         ↓
Lista final de CANDIDATOS VIABLES
```

Lo que un equipo de químicos farmacéuticos hace en **6 meses**, nosotros lo hacemos en **40 minutos**.

---

## La Ventaja Competitiva (10 segundos)

| Ellos | Nosotros |
|-------|----------|
| Un target a la vez | **3 targets simultáneos** |
| Screening manual | **887 features automatizados** |
| Filtros manuales en Excel | **4 filtros ADMET integrados** |
| Semanas de análisis | **40 minutos end-to-end** |
| Sin trazabilidad | **Reportes automáticos por fase** |

---

## El Mercado (10 segundos)

- **Mercado global de antidepresivos:** $18.3B USD (2025), CAGR 3.2%
- **Drug discovery AI market:** $3.9B (2024) → $11.2B (2028)
- **Costo promedio de drug discovery:** $2.6B por fármaco aprobado
- **Nuestro costo operativo:** ~$0 (open source + datos públicos de ChEMBL)

---

## El Modelo de Negocio

### Fase 1 — SaaS para Pharma (Año 1)
Licenciamos la plataforma a laboratorios farmacéuticos como servicio de screening virtual.
- **Precio:** $50K-500K/año por target
- **Clientes objetivo:** Pharma medianas, biotechs, CROs

### Fase 2 — Pipeline Propio (Año 2-3)
Usamos nuestros propios modelos para identificar candidatos y licenciar las moléculas descubiertas.
- **Revenue:** Licensing fees + royalties (industria estándar: 2-5% de ventas)

### Fase 3 — Multi-Indicación (Año 3+)
Expandimos a ansiedad, TDAH, Parkinson, Alzheimer — mismo motor, nuevos targets.

---

## El Equipo

- **[Tu nombre]** — Desarrollo de IA, arquitectura del sistema
- **[Espacio para químico farmacéutico]** — Validación científica
- **[Espacio para negocio]** — Estrategia comercial y regulatoria

---

## El Ask

> **Buscamos $250K USD** en inversión pre-seed para:
>
> 1. **Validación experimental** de los top 10 candidatos identificados por la IA ($150K)
> 2. **Contratación** de un químico medicinal senior ($60K)
> 3. **Propiedad intelectual** — patentes sobre los candidatos descubiertos ($40K)

---

## El Cierre (5 segundos)

> Cada año que un antidepresivo tarda en llegar al mercado, **millones de personas** siguen sin tratamiento efectivo.
>
> Nuestra IA reduce ese tiempo de **años a minutos** en la fase de descubrimiento.
>
> **No estamos reemplazando el laboratorio. Estamos diciéndole al laboratorio exactamente dónde mirar.**

---

### 📊 Métricas Clave

| KPI | Valor |
|-----|-------|
| Moléculas analizadas | 4,000+ por target |
| Targets simultáneos | 3 (SERT, NAT, DAT) |
| Features por molécula | 887 |
| Tiempo de pipeline | ~40 min |
| Costo de operación | ~$0 (datos públicos) |
| Bugs encontrados y corregidos | 5 (documentados) |

---

*NeuroPharma AI — Descubrimiento de fármacos acelerado por Inteligencia Artificial*
*© 2026 Álvaro Vladimir Ocampo Pulido · C.C. 1.107.078.609 & Paula Leandra Ortega Morales · Todos los derechos reservados*
