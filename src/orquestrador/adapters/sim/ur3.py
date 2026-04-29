from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, Slerp

from orquestrador.config import settings
from orquestrador.domain.models import ActionResult, RobotState, RobotStatus


class UR3:
    def __init__(self, sim: Any, sim_ik: Any):
        self.sim = sim
        self.sim_ik = sim_ik

        self.base = sim.getObject(settings.robot_base)
        self.tip = sim.getObject(settings.tip_path)
        self.target = sim.getObject(settings.target_path)
        self.joints = [sim.getObject(f"{settings.robot_base}/joint", {"index": i}) for i in range(6)]

        self.ik_env = sim_ik.createEnvironment()
        self.ik_group = sim_ik.createGroup(self.ik_env)
        sim_ik.setGroupCalculation(
            self.ik_env,
            self.ik_group,
            sim_ik.method_damped_least_squares,
            0.1,
            100,
        )
        _, self.ik_map, _ = sim_ik.addElementFromScene(
            self.ik_env,
            self.ik_group,
            self.base,
            self.tip,
            self.target,
            sim_ik.constraint_pose,
        )

        self.home_q = self._read_joints()
        self.home_pos = np.array(sim.getObjectPosition(self.tip, -1))
        self.home_ori = np.array(sim.getObjectOrientation(self.tip, -1))
        self._traj: dict[str, Any] | None = None

        self._sequencia: str | None = None
        self._seq_estado: str | None = None
        self._seq_tempo = 0.0
        self._seq_pos_objeto: np.ndarray | None = None
        self._seq_pos_destino: np.ndarray | None = None

    @property
    def status(self) -> RobotStatus:
        if self._sequencia is not None:
            return RobotStatus.SEQUENCE
        if self._traj is not None:
            return RobotStatus.MOVING
        return RobotStatus.IDLE

    def _read_joints(self) -> np.ndarray:
        return np.array([self.sim.getJointPosition(j) for j in self.joints])

    def _set_joints(self, q: np.ndarray) -> None:
        for j, a in zip(self.joints, q):
            self.sim.setJointTargetPosition(j, float(a))

    def _solve_ik(self, pos: np.ndarray, ori_rad: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray | None:
        self.sim.setObjectPosition(self.target, -1, pos.tolist())
        self.sim.setObjectOrientation(self.target, -1, ori_rad.tolist())

        if seed is not None:
            for j, a in zip(self.joints, seed):
                self.sim_ik.setJointPosition(self.ik_env, self.ik_map[j], float(a))
        else:
            self.sim_ik.syncFromSim(self.ik_env, self.ik_map)

        self.sim_ik.setObjectPose(
            self.ik_env,
            self.ik_map[self.target],
            self.sim.getObjectPose(self.target, -1),
        )

        if self.sim_ik.handleGroup(self.ik_env, self.ik_group)[0] == self.sim_ik.result_success:
            return np.array([self.sim_ik.getJointPosition(self.ik_env, self.ik_map[j]) for j in self.joints])
        return None

    def _smooth_path(self, waypoints: list[np.ndarray]) -> list[np.ndarray]:
        if len(waypoints) < 2:
            return waypoints

        pts = np.array(waypoints)
        t_wp = np.linspace(0, 1, len(pts))
        t_smooth = np.linspace(0, 1, settings.n_smooth)
        k = min(3, len(pts) - 1)

        smooth = np.zeros((settings.n_smooth, 6))
        for j in range(6):
            spline = make_interp_spline(t_wp, pts[:, j], k=k)
            smooth[:, j] = spline(t_smooth)

        return [smooth[i] for i in range(settings.n_smooth)]

    def _plan_path(self, pos_end: np.ndarray, ori_end_deg: np.ndarray) -> list[np.ndarray] | None:
        pos0 = np.array(self.sim.getObjectPosition(self.tip, -1))
        ori0_rad = np.array(self.sim.getObjectOrientation(self.tip, -1))
        ori_end_rad = np.deg2rad(ori_end_deg)

        r0 = Rotation.from_euler("xyz", ori0_rad)
        r1 = Rotation.from_euler("xyz", ori_end_rad)
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))

        q0 = self._read_joints()

        for n in [settings.n_waypoints, settings.n_waypoints // 2, 1]:
            path = [q0]
            success = True

            for i in range(1, n + 1):
                t = i / n
                pos_t = pos0 + t * (pos_end - pos0)
                ori_t = slerp(t).as_euler("xyz")

                q = self._solve_ik(pos_t, ori_t, path[-1])
                if q is None:
                    q = self._solve_ik(pos_t, ori_t, q0)
                if q is None:
                    success = False
                    break
                path.append(q)

            if success:
                return self._smooth_path(path)

        return None

    def _start_traj(self, q_or_path: np.ndarray | list[np.ndarray]) -> None:
        path = q_or_path if isinstance(q_or_path, list) else self._smooth_path([self._read_joints(), q_or_path])

        total_dist = float(np.max(np.abs(path[-1] - path[0])))
        duration = max(total_dist / settings.vel_max, settings.duracao_min)
        self._traj = {"path": path, "T": duration, "t": 0.0}

    def get_state(self) -> RobotState:
        pos = np.array(self.sim.getObjectPosition(self.tip, -1))
        ori = np.rad2deg(self.sim.getObjectOrientation(self.tip, -1))
        q = np.rad2deg(self._read_joints())
        return RobotState(position=pos, orientation=ori, joint_angles=q, gripper_closed=False, status=self.status)

    @property
    def ocupado(self) -> bool:
        return self._traj is not None or self._sequencia is not None

    def update(self, garra: Any | None = None) -> tuple[bool, str | None]:
        msg = None

        if self._traj is not None:
            self._traj["t"] += settings.dt
            t_norm = min(self._traj["t"] / self._traj["T"], 1.0)
            s = 0.5 * (1 - np.cos(np.pi * t_norm))

            idx = int(s * (len(self._traj["path"]) - 1))
            idx = min(idx, len(self._traj["path"]) - 1)
            self._set_joints(self._traj["path"][idx])

            if self._traj["t"] >= self._traj["T"]:
                self._set_joints(self._traj["path"][-1])
                self._traj = None

        if self._sequencia is not None and garra is not None:
            msg = self._atualizar_sequencia(garra)

        idle = self._traj is None and self._sequencia is None
        return idle, msg

    def _atualizar_sequencia(self, garra: Any) -> str | None:
        if self._sequencia is None:
            return None

        self._seq_tempo += settings.dt

        if self._sequencia == "pegar":
            if self._seq_estado == "abrindo" and self._seq_tempo >= 0.3:
                garra.abrir()
                self._seq_estado = "aproximando"
                self._seq_tempo = 0.0
                pos_aprox = self._seq_pos_objeto.copy()
                pos_aprox[2] += settings.altura_aproximacao
                if not self._move_to(pos_aprox):
                    self._sequencia = None
                    return "IK falhou ao aproximar"
                return "Aproximando do objeto..."

            if self._seq_estado == "aproximando" and self._traj is None:
                self._seq_estado = "descendo"
                self._seq_tempo = 0.0
                if not self._move_to(self._seq_pos_objeto):
                    self._sequencia = None
                    return "IK falhou ao descer"
                return "Descendo ate objeto..."

            if self._seq_estado == "descendo" and self._traj is None:
                self._seq_estado = "fechando"
                self._seq_tempo = 0.0
                garra.fechar()
                return "Fechando garra..."

            if self._seq_estado == "fechando" and self._seq_tempo >= settings.tempo_garra_fechar:
                self._seq_estado = "subindo"
                self._seq_tempo = 0.0
                pos_subir = self._seq_pos_objeto.copy()
                pos_subir[2] += settings.altura_aproximacao
                if not self._move_to(pos_subir):
                    self._sequencia = None
                    return "IK falhou ao subir"
                return "Subindo com objeto..."

            if self._seq_estado == "subindo" and self._traj is None:
                self._sequencia = None
                self._seq_estado = None
                return "Objeto capturado com sucesso!"

        if self._sequencia == "depositar":
            if self._seq_estado == "indo" and self._traj is None:
                self._seq_estado = "descendo"
                self._seq_tempo = 0.0
                pos_soltar = self._seq_pos_destino.copy()
                pos_soltar[2] += settings.altura_soltar
                if not self._move_to(pos_soltar):
                    self._sequencia = None
                    return "IK falhou ao descer para soltar"
                return "Descendo para soltar no chao..."

            if self._seq_estado == "descendo" and self._traj is None:
                self._seq_estado = "abrindo"
                self._seq_tempo = 0.0
                garra.abrir()
                return "Soltando objeto no chao..."

            if self._seq_estado == "abrindo" and self._seq_tempo >= settings.tempo_garra_abrir:
                self._seq_estado = "recuando"
                self._seq_tempo = 0.0
                pos_recuar = self._seq_pos_destino.copy()
                pos_recuar[2] += settings.altura_aprox_deposito + 0.05
                if not self._move_to(pos_recuar):
                    self._sequencia = None
                    return "IK falhou ao recuar"
                return "Recuando..."

            if self._seq_estado == "recuando" and self._traj is None:
                self._sequencia = None
                self._seq_estado = None
                return "Objeto depositado com sucesso!"

        return None

    def _move_to(self, pos: np.ndarray, ori_deg: np.ndarray | None = None) -> bool:
        if ori_deg is None:
            ori_deg = np.array([-90, 0, 0])

        path = self._plan_path(pos, ori_deg)
        if path is None:
            return False
        self._start_traj(path)
        return True

    def move_to_position(self, x: float, y: float, z: float, high: bool = True) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")

        pos = np.array([x, y, z], dtype=np.float64)
        if high:
            pos[2] = max(pos[2], 0.15)

        ori = np.rad2deg(self.sim.getObjectOrientation(self.tip, -1))
        path = self._plan_path(pos, ori)
        if path is None:
            return ActionResult(False, f"IK falhou para posicao [{x:.3f}, {y:.3f}, {z:.3f}]")

        self._start_traj(path)
        return ActionResult(True, f"Movendo para [{x:.3f}, {y:.3f}, {z:.3f}]")

    def move_to_pose(self, x: float, y: float, z: float, rx: float, ry: float, rz: float) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")

        pos = np.array([x, y, z])
        ori = np.array([rx, ry, rz])
        path = self._plan_path(pos, ori)

        if path is None:
            return ActionResult(False, "IK falhou para pose especificada")

        self._start_traj(path)
        return ActionResult(True, f"Movendo para pose [{x:.3f}, {y:.3f}, {z:.3f}] @ [{rx:.1f}, {ry:.1f}, {rz:.1f}]")

    def move_joint(self, joint: int, delta: float) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")
        if not 0 <= joint <= 5:
            return ActionResult(False, f"Junta invalida: {joint}. Use 0-5.")

        q = self._read_joints()
        q[joint] += np.deg2rad(delta)
        self._start_traj(q)
        return ActionResult(True, f"Girando {settings.juntas[joint]} em {delta} graus")

    def go_home(self) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")

        self.sim.setObjectPosition(self.target, -1, self.home_pos.tolist())
        self.sim.setObjectOrientation(self.target, -1, self.home_ori.tolist())
        self._start_traj(self.home_q)
        return ActionResult(True, "Indo para posicao HOME")

    def iniciar_pegar(self, pos_objeto: np.ndarray) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")

        self._sequencia = "pegar"
        self._seq_estado = "abrindo"
        self._seq_tempo = 0.0
        self._seq_pos_objeto = pos_objeto.copy()
        return ActionResult(True, "Iniciando sequencia de captura...")

    def iniciar_depositar(self, pos_destino: np.ndarray) -> ActionResult:
        if self.ocupado:
            return ActionResult(False, "Robo ocupado")

        pos_destino = np.array(pos_destino, dtype=np.float64)
        pos_destino[2] = max(pos_destino[2], 0.02)

        pos_acima = pos_destino.copy()
        pos_acima[2] += settings.altura_aprox_deposito
        if not self._move_to(pos_acima):
            return ActionResult(False, f"IK falhou para aproximacao do destino. pos_acima={pos_acima}")

        self._sequencia = "depositar"
        self._seq_estado = "indo"
        self._seq_tempo = 0.0
        self._seq_pos_destino = pos_destino.copy()
        return ActionResult(True, "Levando objeto para destino (acima)...")

    def cleanup(self) -> None:
        self.sim_ik.eraseEnvironment(self.ik_env)
