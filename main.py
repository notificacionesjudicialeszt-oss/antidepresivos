"""
🧬 Pipeline QSAR Multi-Agente — Orquestador Principal

Ejecuta las 5 fases del pipeline en secuencia:
    1. DataAgent     → Descarga y cura datos de ChEMBL
    2. FeatureAgent  → Feature engineering (fingerprints + descriptores)
    3. TrainingAgent → Entrenamiento dual IC₅₀ / EC₅₀
    4. OptimizationAgent → Optimización con Optuna
    5. ADMETAgent    → Filtros farmacológicos y selección final

Uso:
    python main.py              # Ejecutar todo el pipeline
    python main.py --fase 1     # Ejecutar solo una fase
    python main.py --desde 3    # Ejecutar desde la fase 3 en adelante
"""

import argparse
import logging
import sys
import time

import config
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.training_agent import TrainingAgent
from agents.optimization_agent import OptimizationAgent
from agents.admet_agent import ADMETAgent


# ─── Configurar logging global ───
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


# ─── Mapa de fases ───
AGENTS = {
    1: ("📥 Fase 1: Adquisición de Datos", DataAgent),
    2: ("🧬 Fase 2: Feature Engineering", FeatureAgent),
    3: ("🧠 Fase 3: Entrenamiento Dual", TrainingAgent),
    4: ("⚡ Fase 4: Optimización Optuna", OptimizationAgent),
    5: ("💊 Fase 5: Filtro ADMET", ADMETAgent),
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║     🧬  Pipeline QSAR Multi-Agente · Antidepresivos     ║
║   Predicción de Inhibidores del Transportador SERT       ║
╠══════════════════════════════════════════════════════════╣
║  Target: CHEMBL228 (Transportador de Serotonina)         ║
║  Método: XGBoost + ECFP4 + 43 Descriptores              ║
║  Paper:  Universidad Icesi                               ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_pipeline(fase_unica=None, desde_fase=1):
    """
    Ejecuta el pipeline.

    Args:
        fase_unica: Si se especifica, ejecuta solo esa fase.
        desde_fase: Fase desde la cual iniciar (1-5).
    """
    print_banner()

    start_total = time.time()
    results = {}

    # Determinar qué fases ejecutar
    if fase_unica:
        fases = [fase_unica]
    else:
        fases = range(desde_fase, 6)

    for fase_num in fases:
        if fase_num not in AGENTS:
            logger.error(f"❌ Fase {fase_num} no existe. Fases válidas: 1-5")
            sys.exit(1)

        nombre, AgentClass = AGENTS[fase_num]

        print(f"\n{'═' * 60}")
        print(f"  {nombre}")
        print(f"{'═' * 60}\n")

        agent = AgentClass()
        result = agent.run()
        results[fase_num] = result

        if not result.success:
            logger.error(f"\n❌ Pipeline DETENIDO en: {nombre}")
            logger.error(f"   Errores: {result.errors}")
            print(f"\n💡 Tip: Revisa los logs arriba y corrige el problema.")
            print(f"   Luego ejecuta: python main.py --desde {fase_num}")
            sys.exit(1)

    # ── Resumen final ──
    total_time = time.time() - start_total
    print(f"\n{'═' * 60}")
    print(f"  🎯 PIPELINE COMPLETO")
    print(f"{'═' * 60}")
    print(f"\n  Duración total: {total_time:.1f}s\n")

    for fase_num, result in results.items():
        nombre = AGENTS[fase_num][0]
        print(f"  {result} — {nombre}")

    print(f"\n  📁 Datasets: {config.DATASETS_DIR}")
    print(f"  🧠 Modelos:  {config.MODELOS_DIR}")
    print(f"  📊 Reportes: {config.REPORTES_DIR}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline QSAR Multi-Agente para predicción de antidepresivos"
    )
    parser.add_argument(
        "--fase", type=int, default=None,
        help="Ejecutar solo una fase específica (1-5)"
    )
    parser.add_argument(
        "--desde", type=int, default=1,
        help="Ejecutar desde esta fase en adelante (1-5)"
    )
    args = parser.parse_args()

    run_pipeline(fase_unica=args.fase, desde_fase=args.desde)


if __name__ == "__main__":
    main()
