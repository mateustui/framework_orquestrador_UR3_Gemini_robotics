from __future__ import annotations

import cv2
import numpy as np

from orquestrador.adapters.sim.gripper import Garra
from orquestrador.adapters.sim.ur3 import UR3
from orquestrador.config import settings
from orquestrador.services.orchestrator import LLMOrchestrator


def put_text_with_outline(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    outline_color: tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 3,
) -> None:
    x, y = pos
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        outline_color,
        thickness + outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay_sensor1(
    frame: np.ndarray,
    robot: UR3,
    garra: Garra,
    orchestrator: LLMOrchestrator,
    fps: float = 0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    state = robot.get_state()
    pos, ori, q = state.position, state.orientation, state.joint_angles

    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    orange = (0, 165, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    red = (0, 0, 255)
    black = (0, 0, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 70), black, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if robot._sequencia is not None:
        status, color = ("SEQ", orange)
    elif robot._traj is not None:
        status, color = ("MOVENDO", yellow)
    elif orchestrator.busy:
        status, color = ("EXECUTANDO", cyan)
    else:
        status, color = ("PRONTO", green)

    put_text_with_outline(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if garra.disponivel:
        garra_txt = "GARRA: FECHADA" if garra.fechada else "GARRA: ABERTA"
        garra_cor = red if garra.fechada else green
    else:
        garra_txt = "GARRA: N/A"
        garra_cor = magenta

    put_text_with_outline(frame, garra_txt, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, garra_cor, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), black, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    put_text_with_outline(frame, "POSE", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan, 2)
    put_text_with_outline(
        frame,
        f"{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}",
        (70, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        white,
        2,
    )
    put_text_with_outline(
        frame,
        f"{ori[0]:+.0f} {ori[1]:+.0f} {ori[2]:+.0f}",
        (260, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        yellow,
        2,
    )

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (w - 75, 0), (w, 140), black, -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    put_text_with_outline(frame, "JUNTAS", (w - 68, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, magenta, 2)
    for i in range(6):
        put_text_with_outline(
            frame,
            f"q{i}:{q[i]:+4.0f}",
            (w - 70, 42 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            white,
            1,
        )

    fps_color = green if fps >= 18 else magenta
    put_text_with_outline(
        frame,
        f"FPS: {fps:.0f}",
        (w - 85, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        fps_color,
        2,
    )

    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (w - 70, h - 60), (w, h - 42), black, -1)
    cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
    put_text_with_outline(frame, "CAM 1", (w - 62, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan, 2)

    return frame


def draw_overlay_sensor2(frame: np.ndarray, deteccoes: dict[str, tuple[int, int]]) -> np.ndarray:
    h, w = frame.shape[:2]

    purple = (200, 120, 255)
    text_color = (255, 255, 255)
    bg_color = (30, 30, 30)
    cyan = (255, 255, 0)
    black = (0, 0, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 70, h - 30), (w, h), black, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    put_text_with_outline(frame, "CAM 2", (w - 62, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan, 2)

    if not deteccoes:
        return frame

    scale_x = w / settings.camera_width
    scale_y = h / settings.camera_height

    for label, (px, py) in deteccoes.items():
        px_scaled = int(px * scale_x)
        py_scaled = int(py * scale_y)

        x_size = 8
        thickness = 2
        cv2.line(
            frame,
            (px_scaled - x_size, py_scaled - x_size),
            (px_scaled + x_size, py_scaled + x_size),
            black,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (px_scaled - x_size, py_scaled + x_size),
            (px_scaled + x_size, py_scaled - x_size),
            black,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (px_scaled - x_size, py_scaled - x_size),
            (px_scaled + x_size, py_scaled + x_size),
            purple,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (px_scaled - x_size, py_scaled + x_size),
            (px_scaled + x_size, py_scaled - x_size),
            purple,
            thickness,
            cv2.LINE_AA,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness_text = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness_text)

        text_x = max(4, min(px_scaled - text_w // 2, w - text_w - 4))
        text_y = max(text_h + 4, min(py_scaled - 15, h - 4))

        padding = 3
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - padding, text_y - text_h - padding),
            (text_x + text_w + padding, text_y + padding),
            bg_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        put_text_with_outline(frame, label, (text_x, text_y), font, font_scale, text_color, thickness_text)

    count_text = f"{len(deteccoes)} obj"
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (70, 25), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    put_text_with_outline(frame, count_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)

    return frame
