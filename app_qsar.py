import streamlit as st
import numpy as np
import pandas as pd
import io
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors, AllChem, Fragments

# ─── Configuración de la página ───
st.set_page_config(
    page_title="Análisis In Silico · Antidepresivos",
    page_icon="🧬",
    layout="wide",
)

# ─── CSS personalizado ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        margin-bottom: 0.8rem;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-unit {
        color: #777;
        font-size: 0.75rem;
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-card {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #ccc;
    }
    .pass-badge {
        display: inline-block;
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .fail-badge {
        display: inline-block;
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .bbb-pass {
        background: linear-gradient(135deg, rgba(46,204,113,0.1), rgba(46,204,113,0.05));
        border: 1px solid rgba(46,204,113,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .bbb-fail {
        background: linear-gradient(135deg, rgba(231,76,60,0.1), rgba(231,76,60,0.05));
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stImage"] { display: flex; justify-content: center; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }

    /* Fingerprint heatmap strip */
    .fp-strip {
        display: flex;
        gap: 1px;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }
    .fp-bit-on {
        width: 5px; height: 5px;
        background: #667eea;
        border-radius: 1px;
    }
    .fp-bit-off {
        width: 5px; height: 5px;
        background: rgba(255,255,255,0.04);
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Funciones auxiliares ───
def badge(passed: bool) -> str:
    return '<span class="pass-badge">✓ Cumple</span>' if passed else '<span class="fail-badge">✗ No cumple</span>'

def calc_bbb_score(mw, log_p, tpsa, hbd, rot_bonds):
    """Estimación simplificada de penetración de Barrera Hematoencefálica (Clark 2003)."""
    score = 0
    if mw <= 400: score += 1
    if 1 <= log_p <= 3: score += 1
    if tpsa <= 90: score += 1
    if hbd <= 3: score += 1
    if rot_bonds <= 8: score += 1
    return score


# ─── Antidepresivos SSRI de referencia ───
SSRI_REFERENCE = {
    "Fluoxetina":   "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "Sertralina":   "CNC1CCC(C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
    "Paroxetina":   "C1CNCC(C1C2=CC=C(C=C2)F)COC3=CC4=C(C=C3)OCO4",
    "Citalopram":   "CN(C)CCCC1(OCC2=CC(=CC=C21)C#N)C3=CC=C(C=C3)F",
    "Escitalopram": "CN(C)CCC[C@@]1(OCC2=CC(=CC=C21)C#N)C3=CC=C(C=C3)F",
}


# ═══════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════
st.markdown('<p class="main-title">🧬 Análisis In Silico de Antidepresivos</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ingresa un código SMILES para un análisis molecular completo con RDKit</p>', unsafe_allow_html=True)

st.markdown("---")

# ─── Entrada ───
col_input, col_btn = st.columns([3, 1], gap="medium")

with col_input:
    smiles_input = st.text_input(
        "🔬  Código SMILES de la molécula",
        value="CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        help="Representación SMILES. Por defecto: Fluoxetina (Prozac)",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analizar = st.button("🚀  Analizar Molécula")

st.markdown("---")


# ═══════════════════════════════════════════
#  ANÁLISIS PRINCIPAL
# ═══════════════════════════════════════════
if analizar:
    if not smiles_input.strip():
        st.error("⚠️ Ingresa un código SMILES válido.")
        st.stop()

    mol = Chem.MolFromSmiles(smiles_input.strip())
    if mol is None:
        st.error("❌ No se pudo interpretar el SMILES. Verifica la sintaxis.")
        st.stop()

    # ─── Calcular TODAS las propiedades ───
    mw          = Descriptors.MolWt(mol)
    log_p       = Descriptors.MolLogP(mol)
    tpsa        = Descriptors.TPSA(mol)
    hbd         = Descriptors.NumHDonors(mol)
    hba         = Descriptors.NumHAcceptors(mol)
    rot_bonds   = Descriptors.NumRotatableBonds(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    frac_sp3    = Descriptors.FractionCSP3(mol)
    mr          = Descriptors.MolMR(mol)         # Refractivity molar
    num_rings   = Descriptors.RingCount(mol)
    arom_rings  = Descriptors.NumAromaticRings(mol)
    formal_chg  = Chem.GetFormalCharge(mol)
    formula     = rdMolDescriptors.CalcMolFormula(mol)

    # Grupos funcionales clave para antidepresivos
    fr_amine    = Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol) + Fragments.fr_NH0(mol)
    fr_halogen  = Fragments.fr_halogen(mol)
    fr_ether    = Fragments.fr_ether(mol)
    fr_benzene  = Fragments.fr_benzene(mol)

    # Huella dactilar ECFP4
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_bits = list(fp)
    bits_on = sum(fp_bits)

    # Imagen 2D
    img = Draw.MolToImage(mol, size=(400, 300))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    # BBB Score
    bbb = calc_bbb_score(mw, log_p, tpsa, hbd, rot_bonds)


    # ═══════════════════════════════════════
    #  SECCIÓN 1 — Estructura + Identidad
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">🧪  Estructura e Identidad Molecular</p>', unsafe_allow_html=True)

    c_img, c_info = st.columns([1, 1], gap="large")

    with c_img:
        st.image(buf, caption="Estructura 2D", use_container_width=True)

    with c_info:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fórmula Molecular</div>
            <div class="metric-value" style="font-size:1.5rem;">{formula}</div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peso Molecular</div>
                <div class="metric-value">{mw:.2f}</div>
                <div class="metric-unit">Da (Daltons)</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Lipofilicidad (LogP)</div>
                <div class="metric-value">{log_p:.2f}</div>
                <div class="metric-unit">Coef. de partición</div>
            </div>
            """, unsafe_allow_html=True)

        r3, r4 = st.columns(2)
        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Átomos Pesados</div>
                <div class="metric-value">{heavy_atoms}</div>
                <div class="metric-unit">sin hidrógenos</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Carga Formal</div>
                <div class="metric-value">{formal_chg:+d}</div>
                <div class="metric-unit">neta</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


    # ═══════════════════════════════════════
    #  SECCIÓN 2 — Descriptores Fisicoquímicos
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">📊  Descriptores Fisicoquímicos</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🧲 TPSA", f"{tpsa:.1f} Å²")
    m2.metric("🔄 Enlaces Rotables", int(rot_bonds))
    m3.metric("🤝 Donadores H", int(hbd))
    m4.metric("🎯 Aceptores H", int(hba))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("💎 Refractividad Molar", f"{mr:.1f}")
    m6.metric("⚗️ Fracción sp³", f"{frac_sp3:.2f}")
    m7.metric("💍 Anillos Totales", int(num_rings))
    m8.metric("🔴 Anillos Aromáticos", int(arom_rings))

    st.markdown("---")


    # ═══════════════════════════════════════
    #  SECCIÓN 3 — Grupos Funcionales
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">🔬  Grupos Funcionales Clave</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem;">
        <div class="metric-card">
            <div class="metric-label">Aminas (N-H)</div>
            <div class="metric-value" style="font-size:1.5rem;">{fr_amine}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Halógenos</div>
            <div class="metric-value" style="font-size:1.5rem;">{fr_halogen}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Éteres (-O-)</div>
            <div class="metric-value" style="font-size:1.5rem;">{fr_ether}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Bencenos</div>
            <div class="metric-value" style="font-size:1.5rem;">{fr_benzene}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


    # ═══════════════════════════════════════
    #  SECCIÓN 4 — Regla de Lipinski + BBB
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">💊  Evaluación de Druglikeness</p>', unsafe_allow_html=True)

    col_lip, col_bbb = st.columns(2, gap="large")

    with col_lip:
        st.markdown("##### Regla de Lipinski (Ro5)")
        rules = [
            ("Peso Molecular ≤ 500 Da", mw <= 500, f"{mw:.1f} Da"),
            ("LogP ≤ 5", log_p <= 5, f"{log_p:.2f}"),
            ("Donadores H ≤ 5", hbd <= 5, str(int(hbd))),
            ("Aceptores H ≤ 10", hba <= 10, str(int(hba))),
        ]
        for name, passed, val in rules:
            st.markdown(f'{badge(passed)}  **{name}** — `{val}`', unsafe_allow_html=True)

        lip_pass = sum(p for _, p, _ in rules)
        if lip_pass == 4:
            st.success(f"✅ Cumple **{lip_pass}/4** reglas → Alta biodisponibilidad oral.")
        elif lip_pass >= 3:
            st.warning(f"⚠️ Cumple **{lip_pass}/4** reglas → Aceptable con precaución.")
        else:
            st.error(f"❌ Cumple **{lip_pass}/4** reglas → Baja biodisponibilidad oral.")

    with col_bbb:
        st.markdown("##### Barrera Hematoencefálica (BBB)")
        st.markdown(
            '<div class="info-card">Estimación simplificada basada en Clark 2003. '
            'Crítica para antidepresivos que deben alcanzar el SNC.</div>',
            unsafe_allow_html=True,
        )
        bbb_rules = [
            ("MW ≤ 400", mw <= 400),
            ("1 ≤ LogP ≤ 3", 1 <= log_p <= 3),
            ("TPSA ≤ 90 Å²", tpsa <= 90),
            ("Donadores H ≤ 3", hbd <= 3),
            ("Enlaces Rotables ≤ 8", rot_bonds <= 8),
        ]
        for name, passed in bbb_rules:
            st.markdown(f'{badge(passed)}  {name}', unsafe_allow_html=True)

        css_class = "bbb-pass" if bbb >= 4 else "bbb-fail"
        verdict = "🟢 Alta probabilidad de cruzar la BBB" if bbb >= 4 else ("🟡 Penetración moderada" if bbb >= 3 else "🔴 Baja penetración de BBB")
        st.markdown(f'<div class="{css_class}"><strong>{verdict}</strong> ({bbb}/5)</div>', unsafe_allow_html=True)

        # Regla de Veber
        st.markdown("")
        st.markdown("##### Regla de Veber")
        veber_tpsa = tpsa <= 140
        veber_rot  = rot_bonds <= 10
        st.markdown(f'{badge(veber_tpsa)}  TPSA ≤ 140 Å² — `{tpsa:.1f}`', unsafe_allow_html=True)
        st.markdown(f'{badge(veber_rot)}  Enlaces Rotables ≤ 10 — `{int(rot_bonds)}`', unsafe_allow_html=True)

    st.markdown("---")


    # ═══════════════════════════════════════
    #  SECCIÓN 5 — Huella Dactilar ECFP4
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">🧬  Huella Dactilar Molecular (ECFP4 — 2048 bits)</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card">Representación binaria de la estructura. '
        'Cada bit activo (morado) representa un sub-fragmento químico detectado. '
        f'<strong>{bits_on}</strong> bits activos de 2048 ({bits_on/2048*100:.1f}%).</div>',
        unsafe_allow_html=True,
    )

    # Visualizar los primeros 512 bits como una tira de colores
    fp_html = '<div class="fp-strip">'
    for b in fp_bits[:512]:
        fp_html += f'<div class="fp-bit-{"on" if b else "off"}"></div>'
    fp_html += '</div>'
    st.markdown(fp_html, unsafe_allow_html=True)
    st.caption("Visualización de los primeros 512 de 2048 bits")

    st.markdown("---")


    # ═══════════════════════════════════════
    #  SECCIÓN 6 — Comparación con SSRIs
    # ═══════════════════════════════════════
    st.markdown('<p class="section-header">📈  Comparación vs. SSRIs de Referencia</p>', unsafe_allow_html=True)

    rows = []
    for name, smi in SSRI_REFERENCE.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        rows.append({
            "Fármaco": name,
            "MW (Da)": round(Descriptors.MolWt(m), 1),
            "LogP": round(Descriptors.MolLogP(m), 2),
            "TPSA (Å²)": round(Descriptors.TPSA(m), 1),
            "Don. H": int(Descriptors.NumHDonors(m)),
            "Acpt. H": int(Descriptors.NumHAcceptors(m)),
            "Rot. Bonds": int(Descriptors.NumRotatableBonds(m)),
            "Anillos Arom.": int(Descriptors.NumAromaticRings(m)),
        })

    # Agregar la molécula del usuario
    rows.append({
        "Fármaco": "⭐ Tu molécula",
        "MW (Da)": round(mw, 1),
        "LogP": round(log_p, 2),
        "TPSA (Å²)": round(tpsa, 1),
        "Don. H": int(hbd),
        "Acpt. H": int(hba),
        "Rot. Bonds": int(rot_bonds),
        "Anillos Arom.": int(arom_rings),
    })

    df_compare = pd.DataFrame(rows)
    st.dataframe(
        df_compare.style.highlight_max(axis=0, subset=df_compare.columns[1:], color="rgba(102,126,234,0.25)")
                        .highlight_min(axis=0, subset=df_compare.columns[1:], color="rgba(118,75,162,0.15)"),
        use_container_width=True,
        hide_index=True,
    )


    # ═══════════════════════════════════════
    #  SECCIÓN 7 — Resumen ejecutivo
    # ═══════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="section-header">📝  Resumen Ejecutivo</p>', unsafe_allow_html=True)

    summary_items = []
    summary_items.append(f"**Fórmula:** {formula} — **MW:** {mw:.2f} Da")
    summary_items.append(f"**Lipinski:** {lip_pass}/4 reglas cumplidas")
    summary_items.append(f"**BBB Score:** {bbb}/5 — {verdict}")

    if frac_sp3 >= 0.25:
        summary_items.append("✅ Fracción sp³ adecuada para complejidad 3D")
    else:
        summary_items.append("⚠️ Fracción sp³ baja — molécula muy plana")

    if fr_amine >= 1:
        summary_items.append("✅ Contiene grupo amina (típico en antidepresivos)")
    else:
        summary_items.append("⚠️ Sin grupo amina detectado")

    for item in summary_items:
        st.markdown(f"- {item}")


# ═══════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#555;font-size:0.8rem;">'
    'Desarrollado con RDKit · Streamlit · Proyecto IA — Análisis In Silico de Antidepresivos'
    '</p>',
    unsafe_allow_html=True,
)
