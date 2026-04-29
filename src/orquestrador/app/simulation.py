from __future__ import annotations

import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from orquestrador.adapters.gui.overlays import draw_overlay_sensor1, draw_overlay_sensor2
from orquestrador.adapters.log_system import LogSystem
from orquestrador.adapters.sim.gripper import Garra
from orquestrador.adapters.sim.ur3 import UR3
from orquestrador.adapters.vision.stereo import StereoVision
from orquestrador.config import settings
from orquestrador.services.orchestrator import LLMOrchestrator

RESET_SIM_CMD = "__reset_simulation__"

_SCENE_OBJECTS = [
    "CuboVermelho",
    "CirculoVermelho",
    "CuboAzul",
    "CirculoAzul",
    "CuboVerde",
    "CirculoVerde",
]


def _log_object_states(sim: Any, logger: Any, label: str) -> None:
    logger.system(f"--- {label} ---")
    for name in _SCENE_OBJECTS:
        try:
            handle = sim.getObject(f"/{name}")
            pos = sim.getObjectPosition(handle, -1)
            ori = sim.getObjectOrientation(handle, -1)
            logger.info(
                f"{name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                f"  ori=[{ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}] rad"
            )
        except Exception as exc:
            logger.warning(f"{name}: nao encontrado ({exc})")


def _resolve_scene_path(scene_path: str) -> str:
    scene = Path(scene_path).expanduser()
    if scene.is_absolute() and scene.exists():
        return str(scene)

    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path.cwd() / scene,
        project_root / scene,
        project_root / "scenes" / scene,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    tested = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Cena '{scene_path}' nao encontrada.\n"
        "Defina SCENE_PATH no .env com caminho absoluto/relativo valido.\n"
        f"Caminhos testados:\n{tested}"
    )


def simulation_thread(cmd_queue: Queue, log_queue: Queue, frame_queue: Queue, running: list[bool]) -> None:
    logger = LogSystem(log_queue)
    sim = None
    robot = None

    try:
        if not settings.google_api_key:
            env_path = Path(__file__).resolve().parents[3] / ".env"
            logger.error("GOOGLE_API_KEY nao definido.")
            logger.error(f"Crie o arquivo: {env_path}")
            logger.error('Adicione: GOOGLE_API_KEY="sua_chave_aqui"')
            logger.error("Reinicie o orquestrador apos salvar o .env.")
            return

        logger.system("Conectando ao CoppeliaSim...")
        client = RemoteAPIClient()
        sim = client.require("sim")
        sim_ik = client.require("simIK")

        scene_full_path = _resolve_scene_path(settings.scene_path)

        def start_runtime() -> tuple[UR3, Garra, StereoVision, LLMOrchestrator]:
            logger.system(f"Carregando cena: {scene_full_path}")
            sim.loadScene(scene_full_path)
            sim.setStepping(True)
            sim.startSimulation()

            robot_local = UR3(sim, sim_ik)
            garra_local = Garra(sim, logger)
            vision_local = StereoVision(sim, logger)
            orchestrator_local = LLMOrchestrator(robot_local, garra_local, vision_local, logger)

            if garra_local.disponivel:
                garra_local.abrir()

            logger.success("Sistema inicializado - Robo pronto!")
            logger.info(f"Garra: {'Disponivel' if garra_local.disponivel else 'N/A'}")
            return robot_local, garra_local, vision_local, orchestrator_local

        def reset_runtime(robot_local: UR3 | None) -> tuple[UR3, Garra, StereoVision, LLMOrchestrator]:
            logger.system("Reset da simulacao solicitado...")
            if robot_local:
                robot_local.cleanup()
            try:
                sim.stopSimulation()
            except Exception:
                pass
            time.sleep(0.1)
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            return start_runtime()

        robot, garra, vision, orchestrator = start_runtime()
        _log_object_states(sim, logger, "CONDICAO INICIAL DOS OBJETOS")

        fps = 0.0
        t_fps = time.perf_counter()
        _experiment_start: float | None = None
        _was_busy = False

        while running[0]:
            t0 = time.perf_counter()

            try:
                while True:
                    cmd_text = cmd_queue.get_nowait()

                    if cmd_text.lower() in ["sair", "exit", "quit"]:
                        running[0] = False
                        break
                    if cmd_text == RESET_SIM_CMD:
                        robot, garra, vision, orchestrator = reset_runtime(robot)
                        _log_object_states(sim, logger, "CONDICAO INICIAL DOS OBJETOS (apos reset)")
                        _experiment_start = None
                        _was_busy = False
                        continue

                    if cmd_text.lower() in ["limpar", "limpar deteccoes", "clear"]:
                        vision.limpar_deteccoes()
                        logger.success("Deteccoes visuais limpas")
                        continue

                    if orchestrator.busy:
                        logger.warning("Aguarde a acao atual terminar...")
                        continue

                    cmd_lower = cmd_text.lower().strip()
                    if cmd_lower in ["estado", "status"]:
                        state = robot.get_state()
                        pos, ori = state.position, state.orientation
                        logger.robot(f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        logger.robot(f"Ori: [{ori[0]:.1f}, {ori[1]:.1f}, {ori[2]:.1f}]")
                        continue

                    logger.info("Processando...")
                    _experiment_start = time.perf_counter()
                    _was_busy = False

                    logger.vision("Capturando cena inicial para planejamento...")
                    vision_result = vision.detectar()

                    parsed = orchestrator.plan_with_vision(cmd_text, vision_result)
                    if parsed is None:
                        logger.error("Nao foi possivel interpretar")
                        _experiment_start = None
                        continue

                    if not parsed.get("entendido", False):
                        logger.error("Comando nao compreendido")
                        _experiment_start = None
                        continue

                    explicacao = parsed.get("explicacao", "")
                    if explicacao:
                        logger.robot(explicacao)

                    acoes = parsed.get("acoes", [])
                    if acoes:
                        # Remove go_home do plano do LLM — é adicionado automaticamente via post_queue
                        acoes_filtradas = [a for a in acoes if a.get("funcao") != "go_home"]
                        logger.action(f"Executando {len(acoes_filtradas)} acao(oes) + retorno home...")
                        orchestrator.queue_actions(acoes_filtradas)
                        orchestrator.queue_post_actions([
                            {"funcao": "go_home", "args": {}},
                        ])
                    else:
                        _experiment_start = None

            except Empty:
                pass

            orch_msg = orchestrator.update()
            if orch_msg:
                logger.action(orch_msg)

            _, seq_msg = robot.update(garra)
            if seq_msg:
                if "sucesso" in seq_msg.lower():
                    logger.success(seq_msg)
                elif "falhou" in seq_msg.lower():
                    logger.error(seq_msg)
                else:
                    logger.robot(seq_msg)

            is_busy = orchestrator.busy
            if _experiment_start is not None and _was_busy and not is_busy:
                elapsed = time.perf_counter() - _experiment_start
                logger.timing(f"Tempo de execucao do experimento: {elapsed:.3f}s")
                _log_object_states(sim, logger, "CONDICAO FINAL DOS OBJETOS")
                _experiment_start = None
            _was_busy = is_busy

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_fps, 0.001))
            t_fps = now

            frame1 = vision.capturar_sensor1()
            frame2 = vision.capturar_sensor2()

            if frame1 is not None and frame2 is not None:
                frame1 = draw_overlay_sensor1(frame1, robot, garra, orchestrator, fps)
                frame2 = draw_overlay_sensor2(frame2, vision.deteccoes_cam2)

                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        break
                frame_queue.put((frame1, frame2))

            sim.step()

            elapsed = time.perf_counter() - t0
            if elapsed < settings.dt:
                time.sleep(settings.dt - elapsed)

    except Exception as exc:
        logger.error(f"Erro fatal: {exc}")
        logger.error(traceback.format_exc())

    finally:
        if robot:
            robot.cleanup()
        if sim:
            try:
                sim.stopSimulation()
            except Exception:
                pass
