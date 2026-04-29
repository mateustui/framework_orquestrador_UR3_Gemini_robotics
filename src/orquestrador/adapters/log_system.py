from __future__ import annotations

from queue import Queue

from orquestrador.domain.models import LogLevel


class LogSystem:
    """Centraliza logs para consumo pela GUI."""

    def __init__(self, log_queue: Queue):
        self.queue = log_queue

    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        self.queue.put((message, level.value))

    def info(self, msg: str) -> None:
        self.log(msg, LogLevel.INFO)

    def success(self, msg: str) -> None:
        self.log(msg, LogLevel.SUCCESS)

    def warning(self, msg: str) -> None:
        self.log(msg, LogLevel.WARNING)

    def error(self, msg: str) -> None:
        self.log(msg, LogLevel.ERROR)

    def robot(self, msg: str) -> None:
        self.log(msg, LogLevel.ROBOT)

    def vision(self, msg: str) -> None:
        self.log(msg, LogLevel.VISION)

    def user(self, msg: str) -> None:
        self.log(msg, LogLevel.USER)

    def system(self, msg: str) -> None:
        self.log(msg, LogLevel.SYSTEM)

    def action(self, msg: str) -> None:
        self.log(msg, LogLevel.ACTION)

    def timing(self, msg: str) -> None:
        self.log(msg, LogLevel.TIMING)
