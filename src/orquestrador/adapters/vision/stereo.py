from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from orquestrador.adapters.log_system import LogSystem
from orquestrador.config import settings
from orquestrador.core.geometry import compute_intrinsics, euler_to_rotation, triangulate
from orquestrador.domain.models import ActionResult, Objeto3D
from orquestrador.prompts import PROMPT_DETECTAR_OBJETOS, PROMPT_LOCALIZAR_CAMERA2


_ALIASES_PALAVRAS = {
    "disco": "circulo",
    "redondo": "circulo",
    "bola": "esfera",
}


def _normalizar_nome(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = texto.lower().strip()
    texto = re.sub(r"[_\-]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    palavras = [_ALIASES_PALAVRAS.get(palavra, palavra) for palavra in texto.split()]
    return " ".join(palavras)


class StereoVision:
    def __init__(self, sim: Any, logger: LogSystem):
        self.sim = sim
        self.logger = logger
        self.client_gemini = genai.Client(api_key=settings.google_api_key)

        self.sensor1 = sim.getObject(settings.sensor1_path)
        self.sensor2 = sim.getObject(settings.sensor2_path)

        self.k, self.k_inv = compute_intrinsics(
            settings.camera_width,
            settings.camera_height,
            settings.camera_fov,
        )
        self._calibrar()

        self.objetos: list[Objeto3D] = []
        self.deteccoes_cam2: dict[str, tuple[int, int]] = {}
        self.busy = False

    def _calibrar(self) -> None:
        def get_extrinsics(sensor: int) -> tuple[np.ndarray, np.ndarray]:
            pos = self.sim.getObjectPosition(sensor, -1)
            ori = np.rad2deg(self.sim.getObjectOrientation(sensor, -1))
            rot = euler_to_rotation(ori[0], ori[1], ori[2])
            return rot, np.array(pos)

        self.r1, self.t1 = get_extrinsics(self.sensor1)
        self.r2, self.t2 = get_extrinsics(self.sensor2)

    def _capturar(self, sensor: int) -> np.ndarray:
        img, (w, h) = self.sim.getVisionSensorImg(sensor)
        frame = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.flip(frame, 0)

    def capturar_sensor1(self) -> np.ndarray:
        return self._capturar(self.sensor1)

    def capturar_sensor2(self) -> np.ndarray:
        return self._capturar(self.sensor2)

    def _chamar_gemini(self, frame: np.ndarray, prompt: str) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        resp = self.client_gemini.models.generate_content(
            model=settings.model_id,
            contents=[pil, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        text = resp.text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Resposta invalida do Gemini para deteccao")
        return parsed

    def _parse_deteccoes(self, deteccoes: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        coords: dict[str, tuple[int, int]] = {}
        for d in deteccoes:
            px = int(d["point"][1] * settings.camera_width / 1000)
            py = int(d["point"][0] * settings.camera_height / 1000)
            px = max(0, min(px, settings.camera_width - 1))
            py = max(0, min(py, settings.camera_height - 1))
            coords[str(d["label"])] = (px, py)
        return coords

    def limpar_deteccoes(self) -> None:
        self.deteccoes_cam2 = {}
        self.objetos = []

    def detectar(self) -> ActionResult:
        if self.busy:
            return ActionResult(False, "Deteccao em andamento...")

        self.busy = True
        self.objetos = []
        self.deteccoes_cam2 = {}

        try:
            self.logger.vision("Capturando imagens estereo...")
            img1 = self._capturar(self.sensor1)
            img2 = self._capturar(self.sensor2)

            self.logger.vision("Analisando camera 1...")
            det1 = self._chamar_gemini(img1, PROMPT_DETECTAR_OBJETOS)
            coords1 = self._parse_deteccoes(det1)

            if not coords1:
                return ActionResult(True, "Nenhum objeto detectado", [])

            labels = list(coords1.keys())
            self.logger.vision(f"Correlacionando {len(labels)} objetos na camera 2...")

            prompt2 = PROMPT_LOCALIZAR_CAMERA2.format(labels=labels)
            det2 = self._chamar_gemini(img2, prompt2)
            coords2 = self._parse_deteccoes(det2)
            self.deteccoes_cam2 = coords2.copy()

            for label, px1 in coords1.items():
                if label not in coords2:
                    continue
                px2 = coords2[label]
                pos, erro = triangulate(self.k_inv, self.r1, self.t1, px1, self.r2, self.t2, px2)
                self.objetos.append(
                    Objeto3D(
                        label=label,
                        posicao=pos,
                        erro_mm=erro,
                        pixel_cam1=px1,
                        pixel_cam2=px2,
                    )
                )

            if not self.objetos:
                return ActionResult(True, f"Objetos detectados mas nao triangulados: {labels}", [])

            linhas = [f"{len(self.objetos)} objeto(s) localizado(s):", "-" * 50]
            for i, obj in enumerate(self.objetos, 1):
                p = obj.posicao
                linhas.append(
                    f"  {i}. {obj.label}\n"
                    f"     Posicao: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m\n"
                    f"     Erro estimado: {obj.erro_mm:.1f} mm"
                )
            linhas.append("-" * 50)
            return ActionResult(True, "\n".join(linhas), self.objetos)

        except Exception as exc:
            return ActionResult(False, f"Erro na deteccao: {exc}")
        finally:
            self.busy = False

    def localizar(self, nome: str) -> ActionResult:
        if not self.objetos:
            result = self.detectar()
            if not result.success or not self.objetos:
                return ActionResult(False, result.message or "Nenhum objeto detectado")

        nome_normalizado = _normalizar_nome(nome)
        for obj in self.objetos:
            label_normalizado = _normalizar_nome(obj.label)
            if nome_normalizado in label_normalizado or label_normalizado in nome_normalizado:
                p = obj.posicao
                msg = (
                    f"Objeto encontrado: '{obj.label}'\n"
                    f"   Posicao: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m\n"
                    f"   Erro: {obj.erro_mm:.1f} mm"
                )
                return ActionResult(True, msg, obj.posicao)

        disponiveis = [o.label for o in self.objetos]
        return ActionResult(False, f"'{nome}' nao encontrado.\nDisponiveis: {disponiveis}")
