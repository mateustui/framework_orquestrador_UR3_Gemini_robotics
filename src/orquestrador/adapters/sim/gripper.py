from __future__ import annotations

from typing import Any

from orquestrador.adapters.log_system import LogSystem
from orquestrador.config import settings
from orquestrador.domain.models import ActionResult


class Garra:
    def __init__(self, sim: Any, logger: LogSystem):
        self.sim = sim
        self.logger = logger
        self.signal: str | None = None
        self.fechada = False
        self._configurar()

    def _configurar(self) -> None:
        try:
            connection = self.sim.getObject(f"{settings.robot_base}/connection")
            index = 0
            while True:
                child = self.sim.getObjectChild(connection, index)
                if child == -1:
                    break
                alias = self.sim.getObjectAlias(child, 0)
                if "Gripper" in alias or "gripper" in alias:
                    self.signal = f"{self.sim.getObjectAlias(child, 4)}_close"
                    return
                index += 1
        except Exception:
            self.signal = None

    @property
    def disponivel(self) -> bool:
        return self.signal is not None

    def abrir(self) -> ActionResult:
        if not self.disponivel or self.signal is None:
            return ActionResult(False, "Garra nao disponivel")

        self.sim.clearInt32Signal(self.signal)
        self.fechada = False
        return ActionResult(True, "Garra aberta")

    def fechar(self) -> ActionResult:
        if not self.disponivel or self.signal is None:
            return ActionResult(False, "Garra nao disponivel")

        self.sim.setInt32Signal(self.signal, 1)
        self.fechada = True
        return ActionResult(True, "Garra fechada")
