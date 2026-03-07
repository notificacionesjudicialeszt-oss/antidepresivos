"""
BaseAgent — Clase base abstracta para todos los agentes del pipeline.

Cada agente sigue el ciclo: validate_inputs → execute → validate_outputs → report.
Si una validación falla, el pipeline se detiene con un mensaje claro.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Resultado estandarizado que retorna cada agente."""
    success: bool
    agent_name: str
    data: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    duration_seconds: float = 0.0

    def __str__(self):
        status = "✅" if self.success else "❌"
        return f"{status} [{self.agent_name}] — {self.duration_seconds:.1f}s"


class BaseAgent(ABC):
    """
    Clase base para todos los agentes del pipeline QSAR.

    Subclases deben implementar:
        - validate_inputs(): verificar que los datos de entrada existen
        - execute(): lógica principal del agente
        - validate_outputs(result): verificar que los outputs son correctos
        - generate_report(result): generar reporte de lo que hizo
    """

    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configura logger con formato legible para cada agente."""
        logger = logging.getLogger(self.name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"[%(asctime)s] [{self.name}] %(levelname)s — %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run(self) -> AgentResult:
        """
        Ejecuta el ciclo completo del agente.
        Este método NO debe ser sobreescrito.
        """
        self.logger.info(f"▶ Agente '{self.name}' iniciando...")
        start = time.time()

        try:
            # Paso 1: Validar entradas
            self.logger.info("🔍 Validando inputs...")
            self.validate_inputs()

            # Paso 2: Ejecutar lógica principal
            self.logger.info("⚙️  Ejecutando...")
            data = self.execute()

            # Paso 3: Validar salidas
            self.logger.info("🔍 Validando outputs...")
            self.validate_outputs(data)

            # Paso 4: Generar reporte
            self.logger.info("📝 Generando reporte...")
            self.generate_report(data)

            duration = time.time() - start
            result = AgentResult(
                success=True,
                agent_name=self.name,
                data=data,
                duration_seconds=duration,
            )
            self.logger.info(f"✅ Completado en {duration:.1f}s")
            return result

        except ValidationError as e:
            duration = time.time() - start
            self.logger.error(f"❌ Validación fallida: {e}")
            return AgentResult(
                success=False,
                agent_name=self.name,
                errors=[str(e)],
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            self.logger.error(f"💥 Error inesperado: {e}", exc_info=True)
            return AgentResult(
                success=False,
                agent_name=self.name,
                errors=[str(e)],
                duration_seconds=duration,
            )

    @abstractmethod
    def validate_inputs(self) -> None:
        """
        Verifica que los datos/archivos de entrada existen y son válidos.
        Lanza ValidationError si algo falta.
        """
        ...

    @abstractmethod
    def execute(self) -> dict:
        """
        Lógica principal del agente.
        Retorna un diccionario con los datos/resultados producidos.
        """
        ...

    @abstractmethod
    def validate_outputs(self, data: dict) -> None:
        """
        Verifica que los outputs del agente son correctos.
        Lanza ValidationError si algo no cumple los criterios.
        """
        ...

    @abstractmethod
    def generate_report(self, data: dict) -> None:
        """
        Genera un reporte markdown de lo que hizo el agente.
        Usa el skill report_generator para escritura del archivo.
        """
        ...


class ValidationError(Exception):
    """Error lanzado cuando una validación de input/output falla."""
    pass
