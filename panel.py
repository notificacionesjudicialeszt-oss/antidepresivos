"""
🧬 Panel de Control — Pipeline QSAR Antidepresivos
Panel diseñado para que CUALQUIER persona pueda usar el sistema sin conocimientos técnicos.

Ejecutar con:
    streamlit run panel.py
"""

import io
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config

# ─── Config de página ───
st.set_page_config(
    page_title="Panel QSAR · Antidepresivos",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Rutas del proyecto ───
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELOS_DIR = BASE_DIR / "modelos"
REPORTES_DIR = BASE_DIR / "reportes"


# ═══════════════════════════════════════════════
#  CSS PERSONALIZADO
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ─── Header gradiente ─── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* ─── Cards métricas ─── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .metric-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .metric-card .emoji { font-size: 2rem; margin-bottom: 0.5rem; }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .desc {
        color: #666;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    /* ─── Status badges ─── */
    .status-ready {
        display: inline-block;
        background: rgba(46, 204, 113, 0.12);
        color: #2ecc71;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .status-pending {
        display: inline-block;
        background: rgba(241, 196, 15, 0.12);
        color: #f1c40f;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    .status-missing {
        display: inline-block;
        background: rgba(231, 76, 60, 0.12);
        color: #e74c3c;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }

    /* ─── Phase cards ─── */
    .phase-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .phase-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
    }
    .phase-card .phase-num {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #667eea;
        font-weight: 600;
    }
    .phase-card .phase-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 0.2rem 0;
    }
    .phase-card .phase-desc {
        color: #777;
        font-size: 0.85rem;
    }

    /* ─── Verdict cards ─── */
    .verdict-pass {
        background: linear-gradient(135deg, rgba(46,204,113,0.08), rgba(46,204,113,0.03));
        border: 1px solid rgba(46,204,113,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-fail {
        background: linear-gradient(135deg, rgba(231,76,60,0.08), rgba(231,76,60,0.03));
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* ─── Info cards ─── */
    .info-card {
        background: rgba(102, 126, 234, 0.06);
        border-left: 3px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #bbb;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }

    /* ─── Molecule result card ─── */
    .result-score {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-high {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-medium {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-low {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

    /* ─── Hide streamlit defaults ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════

def check_file_status(path: Path) -> str:
    """Retorna el estado visual de un archivo."""
    if path.exists():
        size_kb = path.stat().st_size / 1024
        if size_kb > 1:
            return "ready"
    return "missing"


def status_badge(status: str, label: str = "") -> str:
    """Genera HTML de badge de estado."""
    if status == "ready":
        return f'<span class="status-ready">✓ {label or "Listo"}</span>'
    elif status == "pending":
        return f'<span class="status-pending">⏳ {label or "En proceso"}</span>'
    else:
        return f'<span class="status-missing">✗ {label or "Pendiente"}</span>'


def count_csv_rows(path: Path) -> int:
    """Cuenta las filas de un CSV."""
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


# ═══════════════════════════════════════════════
#  SIDEBAR — NAVEGACIÓN
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 Pipeline QSAR")
    st.markdown("---")

    page = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "🚀 Ejecutar Pipeline",
            "🔬 Analizar Molécula",
            "📊 Ver Resultados",
            "❓ Ayuda",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Mini status en sidebar — Multi-target
    st.markdown("##### Estado por Target")

    for tkey, tinfo in config.TARGETS.items():
        paths = config.get_target_paths(tkey)
        has_data = check_file_status(paths["dataset_ic50"]) == "ready"
        has_model = check_file_status(paths["modelo_ic50"]) == "ready"
        has_cand = check_file_status(paths["candidatos_viables"]) == "ready"

        icon = "🟢" if has_cand else ("🟡" if has_model else ("🔵" if has_data else "⚪"))
        st.markdown(f"{icon} **{tkey}** — {tinfo['neurotransmitter']}", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PÁGINA: INICIO
# ═══════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Pipeline QSAR</h1>
        <p>Descubre nuevos candidatos a antidepresivos con Inteligencia Artificial</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── ¿Qué es esto? ───
    st.markdown("""
    <div class="info-card">
        <strong>¿Qué hace este sistema?</strong><br><br>
        Analiza <strong>miles de moléculas</strong> de una base de datos científica mundial (ChEMBL)
        para encontrar cuáles tienen mayor probabilidad de funcionar como <strong>antidepresivos</strong>.<br><br>
        Usa <strong>Machine Learning (XGBoost)</strong> para predecir la eficacia de cada molécula
        y luego filtra solo las que son <strong>seguras para el cuerpo humano</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ─── 5 pasos visuales ───
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="emoji">📥</div>
            <div class="label">Fase 1</div>
            <div class="value" style="font-size:1.2rem">Datos</div>
            <div class="desc">Descarga moléculas de ChEMBL</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="emoji">🧬</div>
            <div class="label">Fase 2</div>
            <div class="value" style="font-size:1.2rem">Features</div>
            <div class="desc">Calcula 887 características</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="emoji">🧠</div>
            <div class="label">Fase 3</div>
            <div class="value" style="font-size:1.2rem">Entrenar</div>
            <div class="desc">IA aprende de los datos</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="emoji">⚡</div>
            <div class="label">Fase 4</div>
            <div class="value" style="font-size:1.2rem">Optimizar</div>
            <div class="desc">Ajusta la IA al máximo</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="metric-card">
            <div class="emoji">💊</div>
            <div class="label">Fase 5</div>
            <div class="value" style="font-size:1.2rem">Filtrar</div>
            <div class="desc">Solo candidatos seguros</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # ─── Stats por target ───
    st.markdown("### 🎯 Estado por Target")
    target_cols = st.columns(3)
    for i, (tkey, tinfo) in enumerate(config.TARGETS.items()):
        paths = config.get_target_paths(tkey)
        mol_count = count_csv_rows(paths["dataset_ic50"])
        cand_count = count_csv_rows(paths["candidatos_viables"])
        model_ok = paths["modelo_ic50"].exists() or paths["modelo_ic50_optuna"].exists()

        with target_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">{"�" if tkey=="SERT" else ("🟠" if tkey=="NAT" else "🟢")}</div>
                <div class="label">{tkey}</div>
                <div class="value">{mol_count if mol_count else "—"}</div>
                <div class="desc">{tinfo['neurotransmitter']}<br>
                    Modelo: {"✅" if model_ok else "—"} · Candidatos: {cand_count if cand_count else "—"}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PÁGINA: EJECUTAR PIPELINE
# ═══════════════════════════════════════════════
elif page == "🚀 Ejecutar Pipeline":
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Ejecutar Pipeline</h1>
        <p>Presiona un botón para ejecutar cada fase — sin comandos, sin terminal</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Selector de target ───
    selected_target = st.selectbox(
        "🎯 Selecciona el target",
        options=list(config.TARGETS.keys()) + ["TODOS"],
        format_func=lambda x: f"{x} — {config.TARGETS[x]['neurotransmitter']}" if x != "TODOS" else "🌐 TODOS (SERT + NAT + DAT)",
        help="Elige sobre cuál transportador ejecutar el pipeline",
    )

    st.markdown("---")

    # ─── Las 5 fases como cards ───
    phases = [
        {
            "num": 1, "icon": "📥", "name": "Descargar Datos",
            "desc": "Descarga miles de moléculas de la base de datos científica ChEMBL y las limpia automáticamente.",
            "time": "5-10 minutos", "output": DATASETS_DIR / "dataset_ic50_limpio.csv",
            "detail": "Se conecta a ChEMBL, descarga ~4,000 registros del transportador de serotonina, valida cada molécula con RDKit y elimina duplicados."
        },
        {
            "num": 2, "icon": "🧬", "name": "Calcular Características",
            "desc": "Convierte cada molécula en un vector numérico de ~887 características que la IA puede entender.",
            "time": "2-5 minutos", "output": DATASETS_DIR / "features_887.csv",
            "detail": "Genera huellas dactilares ECFP4 (2,048 bits), filtra los irrelevantes (~844 bits), y calcula 43 propiedades fisicoquímicas."
        },
        {
            "num": 3, "icon": "🧠", "name": "Entrenar la IA",
            "desc": "La inteligencia artificial aprende a predecir qué moléculas serán efectivas como antidepresivos.",
            "time": "1-3 minutos", "output": MODELOS_DIR / "modelo_ic50.pkl",
            "detail": "Entrena un modelo XGBoost que aprende la relación entre la estructura molecular y la eficacia contra el SERT."
        },
        {
            "num": 4, "icon": "⚡", "name": "Optimizar la IA",
            "desc": "Ajusta 9 parámetros internos para que la IA sea lo más precisa posible. Prueba 50 combinaciones.",
            "time": "10-20 minutos", "output": MODELOS_DIR / "modelo_ic50_optuna.pkl",
            "detail": "Usa Optuna (búsqueda bayesiana) con validación cruzada de 5 folds y 50 trials para encontrar los mejores hiperparámetros."
        },
        {
            "num": 5, "icon": "💊", "name": "Filtrar Candidatos",
            "desc": "Pasa cada molécula por 4 filtros farmacológicos reales. Solo sobreviven las candidatas seguras.",
            "time": "1-2 minutos", "output": REPORTES_DIR / "candidatos_viables.csv",
            "detail": "Filtros: QED (calidad de fármaco), Lipinski (biodisponibilidad), BBB (llega al cerebro), PAINS (sin falsos positivos)."
        },
    ]

    for phase in phases:
        status = check_file_status(phase["output"])
        status_html = status_badge(status)

        with st.container():
            col_info, col_action = st.columns([3, 1])

            with col_info:
                st.markdown(f"""
                <div class="phase-card">
                    <div class="phase-num">FASE {phase['num']}</div>
                    <div class="phase-title">{phase['icon']}  {phase['name']}  {status_html}</div>
                    <div class="phase-desc">{phase['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"ℹ️ Más detalles — Fase {phase['num']}"):
                    st.markdown(phase["detail"])
                    st.markdown(f"**Tiempo estimado:** {phase['time']}")
                    st.markdown(f"**Archivo de salida:** `{phase['output'].name}`")

            with col_action:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    f"▶ Ejecutar Fase {phase['num']}",
                    key=f"run_phase_{phase['num']}",
                    use_container_width=True,
                ):
                    with st.spinner(f"{phase['icon']} Ejecutando Fase {phase['num']}... Esto puede tardar {phase['time']}"):
                        cmd = [sys.executable, "main.py", "--fase", str(phase["num"])]
                        if selected_target != "TODOS":
                            cmd.extend(["--target", selected_target])
                        result = subprocess.run(
                            cmd, capture_output=True, text=True,
                            cwd=str(BASE_DIR), encoding="utf-8", errors="replace",
                        )
                        if result.returncode == 0:
                            st.success(f"✅ ¡Fase {phase['num']} completada con éxito!")
                            with st.expander("📋 Ver log completo"):
                                st.code(result.stdout, language="text")
                        else:
                            st.error(f"❌ Error en la Fase {phase['num']}")
                            st.code(result.stderr or result.stdout, language="text")
                    st.rerun()

    st.markdown("---")

    # ─── Botón para ejecutar TODO ───
    st.markdown("### 🎯 Ejecutar Todo el Pipeline")
    st.markdown("""
    <div class="info-card">
        Esto ejecuta las 5 fases en secuencia. Si una fase ya tiene datos, los
        sobreescribirá con datos frescos. Tiempo total estimado: <strong>20-40 minutos</strong>.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 EJECUTAR PIPELINE COMPLETO", use_container_width=True):
        with st.spinner("⏳ Ejecutando pipeline completo... Esto puede tardar 20-40 minutos"):
            cmd = [sys.executable, "main.py"]
            if selected_target != "TODOS":
                cmd.extend(["--target", selected_target])
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(BASE_DIR), encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                st.success("🎉 ¡Pipeline completado exitosamente!")
                st.balloons()
            else:
                st.error("❌ El pipeline se detuvo. Revisa el log:")
            with st.expander("📋 Ver log completo"):
                st.code(result.stdout + "\n" + result.stderr, language="text")
        st.rerun()


# ═══════════════════════════════════════════════
#  PÁGINA: ANALIZAR MOLÉCULA
# ═══════════════════════════════════════════════
elif page == "🔬 Analizar Molécula":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Analizar Molécula</h1>
        <p>Ingresa el código SMILES de cualquier compuesto y obtén un análisis completo</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Referencia rápida ───
    st.markdown("##### 💡 Moléculas de ejemplo (haz clic para copiar)")
    ref_cols = st.columns(5)
    examples = {
        "Fluoxetina": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "Sertralina": "CNC1CCC(C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
        "Paroxetina": "C1CNCC(C1C2=CC=C(C=C2)F)COC3=CC4=C(C=C3)OCO4",
        "Citalopram": "CN(C)CCCC1(OCC2=CC(=CC=C21)C#N)C3=CC=C(C=C3)F",
        "Aspirina": "CC(=O)OC1=CC=CC=C1C(=O)O",
    }
    for i, (name, smi) in enumerate(examples.items()):
        with ref_cols[i]:
            st.code(smi, language=None)
            st.caption(name)

    st.markdown("---")

    # ─── Input ───
    smiles_input = st.text_input(
        "🧪 Código SMILES de la molécula",
        value="CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        help="Es un código estándar que representa la estructura de una molécula. Copia uno de los ejemplos de arriba.",
    )

    if st.button("🔬 Analizar", use_container_width=True):
        if not smiles_input.strip():
            st.error("⚠️ Escribe o pega un código SMILES arriba.")
            st.stop()

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
        except ImportError:
            st.error("❌ RDKit no está instalado. Ejecuta: pip install rdkit")
            st.stop()

        mol = Chem.MolFromSmiles(smiles_input.strip())
        if mol is None:
            st.error("❌ El código SMILES no es válido. Verifica que esté bien escrito.")
            st.stop()

        # ── Calcular propiedades ──
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = Descriptors.qed(mol)
        rot = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)

        # BBB score — Sistema de 5 criterios (Clark 2003)
        bbb_score = 0
        if mw <= 400: bbb_score += 1
        if 1 <= logp <= 3: bbb_score += 1
        if tpsa <= 90: bbb_score += 1
        if hbd <= 3: bbb_score += 1
        if rot <= 8: bbb_score += 1
        bbb_penetra = bbb_score >= 4

        # Lipinski
        lip_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        # PAINS check
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(pains_params)
        pains_matches = pains_catalog.GetMatches(mol)
        pains_free = len(pains_matches) == 0

        # Imagen 2D
        img = Draw.MolToImage(mol, size=(400, 300))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        st.markdown("---")

        # ═══ Resultados ═══

        # ── Score general ──
        score = 0
        score_max = 5
        if qed >= 0.5: score += 1
        if lip_violations <= 1: score += 1
        if bbb_penetra: score += 1
        if pains_free: score += 1
        if 3 <= mw / 100 <= 5: score += 1

        score_pct = score / score_max * 100
        score_class = "score-high" if score >= 4 else ("score-medium" if score >= 3 else "score-low")
        verdict_class = "verdict-pass" if score >= 4 else "verdict-fail"
        verdict_text = (
            "✅ Candidato Prometedor" if score >= 4
            else ("⚠️ Candidato con Observaciones" if score >= 3
            else "❌ No Recomendado")
        )

        # ── Layout principal ──
        col_img, col_score = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("##### 🧪 Estructura Molecular")
            st.image(buf, caption=formula, use_container_width=True)

        with col_score:
            st.markdown("##### 📋 Veredicto")
            st.markdown(f"""
            <div class="{verdict_class}">
                <div class="result-score {score_class}">{score}/{score_max}</div>
                <div style="font-size:1.3rem; font-weight:700; margin-bottom:0.5rem;">{verdict_text}</div>
                <div style="color:#888; font-size:0.85rem;">
                    Basado en QED, Lipinski, BBB, PAINS y peso molecular.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Propiedades con explicación ──
        st.markdown("##### 📊 Propiedades de la Molécula")
        st.markdown("""
        <div class="info-card">
            Cada propiedad determina si la molécula puede funcionar como medicamento.
            <strong>Verde = bueno</strong>, <strong>Rojo = problema</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            qed_status = "✅" if qed >= 0.5 else "⚠️"
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">💊</div>
                <div class="label">Calidad de Fármaco (QED)</div>
                <div class="value">{qed:.2f}</div>
                <div class="desc">{qed_status} {"Buena calidad" if qed >= 0.5 else "Baja calidad"} (mínimo: 0.50)</div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            lip_status = "✅" if lip_violations <= 1 else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">💧</div>
                <div class="label">Biodisponibilidad (Lipinski)</div>
                <div class="value">{4-lip_violations}/4</div>
                <div class="desc">{lip_status} {lip_violations} violaciones {"(aceptable)" if lip_violations <= 1 else "(no recomendable)"}</div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            bbb_status = "✅" if bbb_penetra else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">🧠</div>
                <div class="label">Penetra el Cerebro (BBB)</div>
                <div class="value">{bbb_score}/5</div>
                <div class="desc">{bbb_status} {"Sí penetra" if bbb_penetra else "Baja penetración"} (mínimo: 4/5)</div>
            </div>
            """, unsafe_allow_html=True)

        with p4:
            pains_status = "✅" if pains_free else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">🛡️</div>
                <div class="label">Interferencia (PAINS)</div>
                <div class="value">{"Limpio" if pains_free else "Alerta"}</div>
                <div class="desc">{pains_status} {"Sin interferencias" if pains_free else f"{len(pains_matches)} alertas detectadas"}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Detalles físicoquímicos ──
        st.markdown("##### 🔢 Detalles Técnicos")
        details_df = pd.DataFrame({
            "Propiedad": ["Peso Molecular", "Lipofilicidad (LogP)", "Superficie Polar (TPSA)", "Donadores de Hidrógeno", "Aceptores de Hidrógeno", "Enlaces Rotables", "Anillos", "Fórmula"],
            "Valor": [f"{mw:.1f} Da", f"{logp:.2f}", f"{tpsa:.1f} Å²", str(int(hbd)), str(int(hba)), str(int(rot)), str(int(rings)), formula],
            "¿Qué significa?": [
                "Tamaño de la molécula. Ideal: < 500",
                "Qué tan bien se disuelve en grasa. Ideal: 0-5",
                "Superficie disponible para unirse al agua. Ideal para cerebro: < 90",
                "Puntos que donan hidrógeno. Ideal: ≤ 5",
                "Puntos que aceptan hidrógeno. Ideal: ≤ 10",
                "Flexibilidad de la molécula. Ideal: ≤ 10",
                "Estructuras circulares en la molécula",
                "Composición atómica exacta",
            ],
        })
        st.dataframe(details_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
#  PÁGINA: VER RESULTADOS
# ═══════════════════════════════════════════════
elif page == "📊 Ver Resultados":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Resultados</h1>
        <p>Explora los datos generados por el pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Selector de target para resultados
    res_target = st.selectbox(
        "🎯 Ver resultados de",
        options=list(config.TARGETS.keys()),
        format_func=lambda x: f"{x} — {config.TARGETS[x]['neurotransmitter']}",
        key="results_target",
    )
    res_paths = config.get_target_paths(res_target)

    tab_datos, tab_candidatos, tab_reportes = st.tabs([
        "📥 Datos Descargados",
        "💊 Candidatos Viables",
        "📄 Reportes",
    ])

    with tab_datos:
        if res_paths["dataset_ic50"].exists():
            df = pd.read_csv(res_paths["dataset_ic50"])
            st.markdown(f"**{len(df):,} moléculas** en el dataset IC₅₀ curado de **{res_target}**")

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Moléculas", f"{len(df):,}")
            col_s2.metric("pActivity Promedio", f"{df['pActivity'].mean():.2f}")
            col_s3.metric("pActivity Rango", f"{df['pActivity'].min():.1f} — {df['pActivity'].max():.1f}")

            st.markdown("")
            st.markdown("##### Vista previa de datos")
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)

            st.download_button(
                f"📥 Descargar dataset {res_target} (CSV)",
                data=df.to_csv(index=False),
                file_name=f"dataset_{res_target.lower()}_ic50.csv",
                mime="text/csv",
            )
        else:
            st.info(f"📭 No hay datos de {res_target}. Ejecuta la **Fase 1** para este target.")

    with tab_candidatos:
        candidatos_path = res_paths["candidatos_viables"]
        if candidatos_path.exists():
            df_cand = pd.read_csv(candidatos_path)
            st.markdown(f"**{len(df_cand):,} candidatos** de **{res_target}** pasaron todos los filtros ADMET")

            st.dataframe(df_cand, use_container_width=True, hide_index=True)

            st.download_button(
                f"📥 Descargar candidatos {res_target} (CSV)",
                data=df_cand.to_csv(index=False),
                file_name=f"candidatos_viables_{res_target.lower()}.csv",
                mime="text/csv",
            )
        else:
            st.info(f"📭 No hay candidatos de {res_target}. Ejecuta la **Fase 5** para este target.")

    with tab_reportes:
        # Filtrar reportes por target
        target_lower = res_target.lower()
        reportes = [r for r in REPORTES_DIR.glob(f"*{target_lower}*.md")] if REPORTES_DIR.exists() else []
        if reportes:
            for reporte in sorted(reportes):
                with st.expander(f"📄 {reporte.stem}"):
                    st.markdown(reporte.read_text(encoding="utf-8"))
        else:
            st.info("📭 Aún no hay reportes. Los reportes se generan al ejecutar cada fase.")


# ═══════════════════════════════════════════════
#  PÁGINA: AYUDA
# ═══════════════════════════════════════════════
elif page == "❓ Ayuda":
    st.markdown("""
    <div class="main-header">
        <h1>❓ Ayuda Rápida</h1>
        <p>Todo lo que necesitas saber para usar este sistema</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🤔 ¿Qué es QSAR?")
    st.markdown("""
    **QSAR** (Quantitative Structure-Activity Relationship) es una técnica que predice
    qué tan efectiva será una molécula como medicamento basándose en su estructura química.

    En lugar de probar cada molécula en un laboratorio (que toma años y millones de dólares),
    usamos **inteligencia artificial** para predecir cuáles vale la pena probar.
    """)

    st.markdown("---")

    st.markdown("### 🧪 ¿Qué es un código SMILES?")
    st.markdown("""
    SMILES es una forma de escribir moléculas como texto. Cada letra y símbolo representa un átomo o enlace:

    | Ejemplo | Molécula |
    |---------|----------|
    | `O` | Agua (H₂O) |
    | `CCO` | Etanol (alcohol) |
    | `CC(=O)OC1=CC=CC=C1C(=O)O` | Aspirina |
    | `CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F` | Fluoxetina (Prozac) |

    Puedes buscar el SMILES de cualquier molécula en [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
    """)

    st.markdown("---")

    st.markdown("### 🚀 ¿Cómo uso el sistema?")
    st.markdown("""
    1. **Ejecutar Pipeline** → Ve a la sección "Ejecutar Pipeline" y presiona los botones en orden (Fase 1 → 2 → 3 → 4 → 5)
    2. **Analizar Molécula** → Pega un código SMILES y obtén un análisis instantáneo de si podría funcionar como antidepresivo
    3. **Ver Resultados** → Explora los datos descargados, los candidatos viables y los reportes generados
    """)

    st.markdown("---")

    st.markdown("### 📊 ¿Qué significan los filtros?")
    st.markdown("""
    | Filtro | ¿Qué evalúa? | ¿Por qué importa? |
    |--------|--------------|-------------------|
    | **QED** | ¿Se parece a un medicamento? | Si es < 0.5, probablemente no funcione como fármaco |
    | **Lipinski** | ¿El cuerpo la absorbe? | Si viola > 1 regla, no se absorbe bien por vía oral |
    | **BBB** | ¿Llega al cerebro? | Los antidepresivos DEBEN llegar al sistema nervioso central |
    | **PAINS** | ¿Da falsos positivos en tests? | Algunas moléculas "engañan" los experimentos |
    """)

    st.markdown("---")

    st.markdown("### ⚠️ Problemas comunes")
    st.markdown("""
    | Problema | Solución |
    |----------|----------|
    | La Fase 1 tarda mucho | Normal — está descargando ~4,000 moléculas de internet |
    | La Fase 4 tarda mucho | Normal — está probando 50 configuraciones distintas |
    | "Error: Optuna no instalado" | Ve a la terminal y escribe: `pip install optuna` |
    | "SMILES no válido" | Verifica el código en [PubChem](https://pubchem.ncbi.nlm.nih.gov/) |
    """)


# ═══════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#444;font-size:0.75rem;">'
    'Pipeline QSAR Multi-Agente · Predicción de Antidepresivos · '
    'Transportador de Serotonina (SERT)'
    '</p>',
    unsafe_allow_html=True,
)
