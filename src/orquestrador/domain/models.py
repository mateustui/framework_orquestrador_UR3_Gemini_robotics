from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class RobotStatus(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    SEQUENCE = "SEQUENCE"
    DETECTING = "DETECTING"
    ERROR = "ERROR"


class LogLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ROBOT = "robot"
    VISION = "vision"
    USER = "user"
    SYSTEM = "system"
    ACTION = "action"
    TIMING = "timing"


@dataclass
class Objeto3D:
    label: str
    posicao: np.ndarray
    erro_mm: float
    pixel_cam1: tuple[int, int] = (0, 0)
    pixel_cam2: tuple[int, int] = (0, 0)


@dataclass
class ActionResult:
    success: bool
    message: str
    data: Any = None


@dataclass
class RobotState:
    position: np.ndarray
    orientation: np.ndarray
    joint_angles: np.ndarray
    gripper_closed: bool
    status: RobotStatus
