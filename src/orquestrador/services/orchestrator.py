from __future__ import annotations

import json
import time
import re
import unicodedata
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from orquestrador.adapters.log_system import LogSystem
from orquestrador.adapters.sim.gripper import Garra
from orquestrador.adapters.sim.ur3 import UR3
from orquestrador.adapters.vision.stereo import StereoVision
from orquestrador.config import settings
from orquestrador.domain.models import ActionResult, Objeto3D
from orquestrador.prompts import ROBOT_API_SCHEMA, ROBOT_API_SCHEMA_WITH_VISION


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


class LLMOrchestrator:
    def __init__(
        self,
        robot: UR3,
        garra: Garra,
        vision: StereoVision,
        logger: LogSystem,
        client: Any | None = None,
    ):
        self.robot = robot
        self.garra = garra
        self.vision = vision
        self.logger = logger
        self.client = client or genai.Client(api_key=settings.google_api_key)

        self._action_queue: list[dict[str, Any]] = []
        self._post_queue: list[dict[str, Any]] = []
        self._waiting = False
        self._wait_until = 0.0
        self._mem_pos: dict[str, np.ndarray] = {}
        self._held_object_name: str | None = None

    def parse_command(self, command: str) -> dict[str, Any] | None:
        try:
            prompt = ROBOT_API_SCHEMA.format(command=command)
            resp = self.client.models.generate_content(
                model=settings.model_id,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_text = (resp.text or "").strip()
            if not raw_text:
                self.logger.warning("Gemini retornou resposta vazia.")
                return None

            text = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                self.logger.warning("Resposta do Gemini nao esta no formato JSON esperado.")
                return None
            self.logger.system("Resposta do Gemini:")
            self.logger.system(json.dumps(parsed, ensure_ascii=False, indent=2))
            return parsed
        except Exception as exc:
            self.logger.error(f"Erro ao interpretar: {exc}")
            return None

    def _build_scene_description(self, vision_result: ActionResult) -> str:
        if not vision_result.success or not vision_result.data:
            return "Nenhum objeto detectado na cena."
        lines = ["Objetos detectados (label: [x, y, z] metros):"]
        for obj in vision_result.data:
            if isinstance(obj, Objeto3D):
                p = obj.posicao
                lines.append(f"  - {obj.label}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
        return "\n".join(lines)

    def plan_with_vision(self, command: str, vision_result: ActionResult) -> dict[str, Any] | None:
        scene_desc = self._build_scene_description(vision_result)
        self.logger.vision(f"Cena inicial registrada:\n{scene_desc}")

        if vision_result.success and vision_result.data:
            for obj in vision_result.data:
                if isinstance(obj, Objeto3D):
                    key = _normalizar_nome(obj.label)
                    self._mem_pos[key] = np.array(obj.posicao, dtype=np.float64)
            self.logger.vision(f"{len(vision_result.data)} posicao(oes) carregada(s) na memoria.")

        try:
            prompt = ROBOT_API_SCHEMA_WITH_VISION.format(
                scene_description=scene_desc,
                command=command,
            )
            resp = self.client.models.generate_content(
                model=settings.model_id,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_text = (resp.text or "").strip()
            if not raw_text:
                self.logger.warning("Gemini retornou resposta vazia.")
                return None
            text = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                self.logger.warning("Resposta do Gemini nao esta no formato JSON esperado.")
                return None
            self.logger.system("Plano gerado pelo Gemini (com visao):")
            self.logger.system(json.dumps(parsed, ensure_ascii=False, indent=2))
            return parsed
        except Exception as exc:
            self.logger.error(f"Erro ao planejar com visao: {exc}")
            return None

    def queue_actions(self, actions: list[dict[str, Any]]) -> None:
        self._action_queue.extend(actions)

    def queue_post_actions(self, actions: list[dict[str, Any]]) -> None:
        self._post_queue.extend(actions)

    def execute_action(self, action: dict[str, Any]) -> ActionResult:
        func_name = str(action.get("funcao", ""))
        args = action.get("args", {})

        try:
            if func_name == "move_to_position":
                return self.robot.move_to_position(
                    args.get("x", 0),
                    args.get("y", 0),
                    args.get("z", 0.2),
                    args.get("high", True),
                )

            if func_name == "move_to_pose":
                return self.robot.move_to_pose(
                    args.get("x", 0),
                    args.get("y", 0),
                    args.get("z", 0.2),
                    args.get("rx", -90),
                    args.get("ry", 0),
                    args.get("rz", 0),
                )

            if func_name == "move_joint":
                return self.robot.move_joint(args.get("joint", 0), args.get("delta", 0))

            if func_name == "go_home":
                return self.robot.go_home()

            if func_name == "open_gripper":
                result = self.garra.abrir()
                if result.success:
                    self._held_object_name = None
                return result

            if func_name == "close_gripper":
                return self.garra.fechar()

            if func_name == "detect_objects":
                result = self.vision.detectar()
                if result.success and result.data:
                    self.logger.vision(result.message)
                return result

            if func_name == "locate_object":
                result = self.vision.localizar(args.get("name", ""))
                if result.success:
                    self.logger.vision(result.message)
                return result

            if func_name == "pick_object":
                obj_name = args.get("object_name", "")
                obj_key = _normalizar_nome(obj_name)
                self.logger.action(f"Localizando '{obj_name}'...")
                det_result = self.vision.detectar()
                if det_result.success and det_result.data:
                    self.logger.vision(det_result.message)
                result = self.vision.localizar(obj_name)
                if not result.success:
                    if obj_key in self._mem_pos:
                        self.logger.warning(
                            f"Visao falhou para '{obj_name}'. Usando posicao em memoria: {self._mem_pos[obj_key]}"
                        )
                        pick_result = self.robot.iniciar_pegar(np.array(self._mem_pos[obj_key], dtype=np.float64))
                        if pick_result.success:
                            self._held_object_name = obj_name
                        return pick_result
                    return result
                self.logger.vision(result.message)
                pick_result = self.robot.iniciar_pegar(result.data)
                if pick_result.success:
                    self._held_object_name = obj_name
                return pick_result

            if func_name == "place_at_position":
                def f(v: Any, default: float) -> float:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return float(default)

                x = f(args.get("x", 0.0), 0.0)
                y = f(args.get("y", 0.0), 0.0)
                z = f(args.get("z", 0.03), 0.03)
                pos = np.array([x, y, z], dtype=np.float64)
                place_result = self.robot.iniciar_depositar(pos)
                if place_result.success and self._held_object_name:
                    self._mem_pos[_normalizar_nome(self._held_object_name)] = np.array(pos, dtype=np.float64)
                    self._held_object_name = None
                return place_result

            if func_name == "place_on_object":
                target_name = args.get("target_name", "")
                self.logger.action(f"Localizando destino '{target_name}'...")
                result = self.vision.localizar(target_name)
                if not result.success:
                    return result
                self.logger.vision(result.message)
                place_result = self.robot.iniciar_depositar(result.data)
                if place_result.success and self._held_object_name:
                    self._mem_pos[_normalizar_nome(self._held_object_name)] = np.array(result.data, dtype=np.float64)
                    self._held_object_name = None
                return place_result

            if func_name == "get_robot_state":
                state = self.robot.get_state()
                pos = state.position
                ori = state.orientation
                q = state.joint_angles

                msg = (
                    "Estado atual do robo:\n"
                    "-------------------------------------\n"
                    f"  Posicao (m):    [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]\n"
                    f"  Orientacao (graus): [{ori[0]:+.1f}, {ori[1]:+.1f}, {ori[2]:+.1f}]\n"
                    "  Juntas (graus):\n"
                    f"     Base:     {q[0]:+.1f}\n"
                    f"     Ombro:    {q[1]:+.1f}\n"
                    f"     Cotovelo: {q[2]:+.1f}\n"
                    f"     Punho1:   {q[3]:+.1f}\n"
                    f"     Punho2:   {q[4]:+.1f}\n"
                    f"     Punho3:   {q[5]:+.1f}\n"
                    "-------------------------------------\n"
                    f"  Garra: {'Fechada' if self.garra.fechada else 'Aberta'}\n"
                    f"  Status: {state.status.value}"
                )
                return ActionResult(True, msg)

            if func_name == "wait":
                seconds = float(args.get("seconds", 1.0))
                self._waiting = True
                self._wait_until = time.time() + seconds
                return ActionResult(True, f"Aguardando {seconds}s...")

            if func_name == "save_object_position":
                name = args.get("name", "")
                key = args.get("key", "") or name
                result = self.vision.localizar(name)
                if not result.success:
                    return result
                self._mem_pos[_normalizar_nome(key)] = np.array(result.data, dtype=np.float64)
                self.logger.vision(f"Posicao salva: {key} = {self._mem_pos[_normalizar_nome(key)]}")
                return ActionResult(True, f"Posicao do '{name}' salva em '{key}'")

            if func_name == "place_at_saved":
                key = args.get("key", "")
                key_normalizada = _normalizar_nome(key)
                if key_normalizada not in self._mem_pos:
                    return ActionResult(False, f"Posicao '{key}' nao encontrada na memoria. Salve primeiro.")
                pos = np.array(self._mem_pos[key_normalizada], dtype=np.float64)
                place_result = self.robot.iniciar_depositar(pos)
                if place_result.success and self._held_object_name:
                    self._mem_pos[_normalizar_nome(self._held_object_name)] = np.array(pos, dtype=np.float64)
                    self._held_object_name = None
                return place_result

            return ActionResult(False, f"Funcao desconhecida: {func_name}")

        except Exception as exc:
            return ActionResult(False, f"Erro ao executar {func_name}: {exc}")

    def update(self) -> str | None:
        if self._waiting:
            if time.time() >= self._wait_until:
                self._waiting = False
            else:
                return None

        if self.robot.ocupado:
            return None

        # Processar fila principal
        if self._action_queue:
            action = self._action_queue.pop(0)
            result = self.execute_action(action)
            func_name = action.get("funcao", "")

            if func_name in ["detect_objects", "locate_object", "get_robot_state"]:
                if result.success:
                    if func_name == "get_robot_state":
                        self.logger.robot(result.message)
                    return f"{func_name} concluido"
                self._action_queue.clear()
                return f"{func_name}: {result.message}"

            if result.success:
                return f"{func_name}: {result.message}"

            self._action_queue.clear()
            return f"{func_name}: {result.message}"

        # Processar pos-acoes (go_home) — executam mesmo apos erros na fila principal
        if self._post_queue:
            action = self._post_queue.pop(0)
            result = self.execute_action(action)
            func_name = action.get("funcao", "")
            return f"{func_name}: {result.message}"

        return None

    @property
    def busy(self) -> bool:
        return bool(self._action_queue) or bool(self._post_queue) or self._waiting or self.robot.ocupado
