"""
🧬 Pipeline QSAR Multi-Agente Multi-Target — Orquestador Principal

Ejecuta las 5 fases del pipeline para uno o todos los targets (SERT, NAT, DAT).

Uso:
    python main.py                          # Todo el pipeline, TODOS los targets
    python main.py --target SERT            # Solo SERT
    python main.py --target NAT --fase 1    # Solo Fase 1 de NAT
    python main.py --fase 1                 # Fase 1 para TODOS los targets
    python main.py --desde 3                # Fases 3-5 para TODOS los targets
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


# ─── Logging global ───
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


# ─── Mapa de fases ───
PHASE_INFO = {
    1: "Adquisicion de Datos",
    2: "Feature Engineering",
    3: "Entrenamiento Dual",
    4: "Optimizacion Optuna",
    5: "Filtro ADMET",
}

PHASE_AGENTS = {
    1: DataAgent,
    2: FeatureAgent,
    3: TrainingAgent,
    4: OptimizationAgent,
    5: ADMETAgent,
}


def print_banner():
    print("""
+----------------------------------------------------------+
|     Pipeline QSAR Multi-Agente - Antidepresivos          |
|   Prediccion de Inhibidores de Transportadores           |
+----------------------------------------------------------+
|  SERT (CHEMBL228) - Transportador de Serotonina          |
|  NAT  (CHEMBL222) - Transportador de Norepinefrina       |
|  DAT  (CHEMBL238) - Transportador de Dopamina            |
+----------------------------------------------------------+
    """)


def run_pipeline(targets=None, fase_unica=None, desde_fase=1):
    """
    Ejecuta el pipeline para uno o más targets.

    Args:
        targets: Lista de target keys (ej: ["SERT", "NAT"]). None = todos.
        fase_unica: Si se especifica, solo esa fase.
        desde_fase: Fase desde la cual iniciar.
    """
    print_banner()

    if targets is None:
        targets = list(config.TARGETS.keys())

    # Determinar fases
    if fase_unica:
        fases = [fase_unica]
    else:
        fases = list(range(desde_fase, 6))

    start_total = time.time()
    all_results = {}

    for target_key in targets:
        if target_key not in config.TARGETS:
            logger.error(f"Target '{target_key}' no existe. Opciones: {list(config.TARGETS.keys())}")
            sys.exit(1)

        target_info = config.TARGETS[target_key]
        print(f"\n{'='*60}")
        print(f"  TARGET: {target_key} — {target_info['name']}")
        print(f"  ChEMBL: {target_info['chembl_id']}")
        print(f"  Neurotransmisor: {target_info['neurotransmitter']}")
        print(f"{'='*60}")

        target_results = {}

        for fase_num in fases:
            AgentClass = PHASE_AGENTS[fase_num]
            fase_name = PHASE_INFO[fase_num]

            print(f"\n  --- Fase {fase_num}: {fase_name} [{target_key}] ---\n")

            agent = AgentClass(target_key=target_key)
            result = agent.run()
            target_results[fase_num] = result

            if not result.success:
                logger.error(f"\n  PIPELINE DETENIDO en Fase {fase_num} [{target_key}]")
                logger.error(f"  Errores: {result.errors}")
                print(f"\n  Tip: Corrige y ejecuta:")
                print(f"       python main.py --target {target_key} --desde {fase_num}")
                sys.exit(1)

        all_results[target_key] = target_results

    # ── Resumen final ──
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETO")
    print(f"{'='*60}")
    print(f"\n  Duracion total: {total_time:.1f}s\n")

    for target_key, results in all_results.items():
        print(f"  [{target_key}]")
        for fase_num, result in results.items():
            fase_name = PHASE_INFO[fase_num]
            status = "OK" if result.success else "FAIL"
            print(f"    {status} Fase {fase_num}: {fase_name} ({result.duration_seconds:.1f}s)")
        print()

    print(f"  Datasets: {config.DATASETS_DIR}")
    print(f"  Modelos:  {config.MODELOS_DIR}")
    print(f"  Reportes: {config.REPORTES_DIR}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline QSAR Multi-Agente para prediccion de antidepresivos"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target especifico: SERT, NAT, DAT. Si no se especifica, corre todos."
    )
    parser.add_argument(
        "--fase", type=int, default=None,
        help="Ejecutar solo una fase especifica (1-5)"
    )
    parser.add_argument(
        "--desde", type=int, default=1,
        help="Ejecutar desde esta fase en adelante (1-5)"
    )
    args = parser.parse_args()

    targets = [args.target] if args.target else None
    run_pipeline(targets=targets, fase_unica=args.fase, desde_fase=args.desde)


if __name__ == "__main__":
    main()
