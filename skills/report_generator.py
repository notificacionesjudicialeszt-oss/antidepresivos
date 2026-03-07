"""
Skill: Report Generator
Genera reportes markdown automáticos para cada fase del pipeline.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_report(
    title: str,
    agent_name: str,
    sections: dict,
    output_path: Path,
    duration_seconds: Optional[float] = None,
) -> Path:
    """
    Genera un reporte markdown con secciones dinámicas.

    Args:
        title: Título del reporte.
        agent_name: Nombre del agente que generó el reporte.
        sections: Diccionario {nombre_seccion: contenido}.
                  El contenido puede ser str, dict, o list.
        output_path: Ruta donde guardar el .md.
        duration_seconds: Duración de la ejecución en segundos.

    Returns:
        Path del archivo generado.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> Generado por **{agent_name}** el {timestamp}")
    if duration_seconds:
        lines.append(f"> Duración: {duration_seconds:.1f} segundos")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sections
    for section_name, content in sections.items():
        lines.append(f"## {section_name}")
        lines.append("")

        if isinstance(content, dict):
            # Renderizar como tabla
            lines.append("| Métrica | Valor |")
            lines.append("|---|---|")
            for key, value in content.items():
                lines.append(f"| {key} | {value} |")
        elif isinstance(content, list):
            # Renderizar como lista
            for item in content:
                lines.append(f"- {item}")
        else:
            # Renderizar como texto
            lines.append(str(content))

        lines.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"📄 Reporte guardado en: {output_path}")

    return output_path
