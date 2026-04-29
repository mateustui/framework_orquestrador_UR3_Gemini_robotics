from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    model_id: str = os.getenv("MODEL_ID", "gemini-robotics-er-1.5-preview")

    scene_path: str = os.getenv("SCENE_PATH", "experimento-ur3.ttt")
    robot_base: str = "/UR3"
    tip_path: str = "/UR3/EFrame"
    target_path: str = "/alvo"
    sensor1_path: str = "/visionSensor1"
    sensor2_path: str = "/visionSensor2"

    dt: float = 0.05
    vel_max: float = 1.0
    duracao_min: float = 1.0
    n_waypoints: int = 5
    n_smooth: int = 50
    juntas: tuple[str, ...] = ("base", "ombro", "cotovelo", "punho1", "punho2", "punho3")

    camera_width: int = 640
    camera_height: int = 480
    camera_fov: int = 90

    display_width: int = 440
    display_height: int = 330

    tempo_garra_fechar: float = 1.5
    tempo_garra_abrir: float = 1.0
    altura_aproximacao: float = 0.05
    altura_soltar: float = 0.01
    altura_aprox_deposito: float = 0.08

    mic_sample_rate: int = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    mic_channels: int = int(os.getenv("MIC_CHANNELS", "1"))
    stt_language: str = os.getenv("STT_LANGUAGE", "pt")
    stt_model_size: str = os.getenv("STT_MODEL_SIZE", "base")
    stt_device: str = os.getenv("STT_DEVICE", "cpu")
    stt_compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8")
    stt_beam_size: int = int(os.getenv("STT_BEAM_SIZE", "1"))
    stt_vad_filter: bool = os.getenv("STT_VAD_FILTER", "false").lower() in {"1", "true", "yes", "on"}


settings = Settings()
