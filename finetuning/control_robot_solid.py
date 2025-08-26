#!/usr/bin/env python3
import sys
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import mujoco
import glfw

from recording import create_lerobot_recorder, add_lerobot_controls


# ---------------------------- Configuration ----------------------------
@dataclass(frozen=True)
class TeleopConfig:
    xml_path: str = \
        "/home/adam/Documents/coding/autonomous/franka_emika_panda/scene_bar_new.xml"
    ee_site_candidates: Tuple[str, ...] = ("ee_site",)
    arm_joint_names: Tuple[str, ...] = ("joint1","joint2","joint3","joint4","joint5","joint6","joint7")
    arm_act_names: Tuple[str, ...] = ("actuator1","actuator2","actuator3","actuator4","actuator5","actuator6","actuator7")
    gripper_act_name: str = "actuator8"
    up_axis: int = 2
    xy_step: float = 0.01
    z_step: float = 0.01
    kp_pos: float = 12.0
    damping_lambda: float = 0.10
    ctrl_hz: float = 60.0
    subst_steps: int = 2
    target_radius: float = 0.01
    target_rgba: Tuple[float, float, float, float] = (1.0, 0.1, 0.1, 1.0)
    camera_orbit_sens: float = 100.0
    camera_pan_sens: float = 100.0
    camera_zoom_sens: float = 60.0
    j1_limit: float = math.radians(10.0)
    j1_limit_k: float = 4.0
    j1_center_k: float = 200.0
    j2_pref: float = -0.5
    j2_pref_k: float = 0.8
    kp_yaw: float = 6.0
    # Joint jogging
    joint_step: float = 0.01  # rad per control tick
    joint_step_coarse_factor: float = 4.0  # when Shift held
    # Sticky grasp config
    grasp_capture_dist: float = 0.15


# ------------------------------ Utilities ------------------------------
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ------------------------------ Input I/O ------------------------------
class InputState:
    def __init__(self):
        self.left=False; self.right=False; self.up=False; self.down=False
        self.z_up=False; self.z_down=False
        self.shift=False
        self.right_drag=False; self.middle_drag=False
        self.last_x=0.0; self.last_y=0.0
        self.grip_close=False; self.grip_open=False
        self.reset=False
        self.save_state=False
        # per-joint jogging flags (+ increases angle, - decreases)
        self.jog_plus = [False]*7
        self.jog_minus = [False]*7

    def reset_oneshot(self):
        self.grip_close=False; self.grip_open=False; self.reset=False; self.save_state=False


class InputController:
    def __init__(self, config: TeleopConfig, model, scene, cam):
        self.cfg = config
        self.inp = InputState()
        self.model = model
        self.scene = scene
        self.cam = cam

    def install(self, window):
        def on_cursor(win, x, y):
            dx = x - self.inp.last_x
            dy = y - self.inp.last_y
            if self.inp.right_drag:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx/self.cfg.camera_orbit_sens, 0, self.scene, self.cam)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, -dy/self.cfg.camera_orbit_sens, self.scene, self.cam)
            elif self.inp.middle_drag:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,   dx/self.cfg.camera_pan_sens, 0, self.scene, self.cam)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V,   0,  -dy/self.cfg.camera_pan_sens, self.scene, self.cam)
            self.inp.last_x, self.inp.last_y = x, y

        def on_button(win, button, action, mods):
            pressed = action == glfw.PRESS
            if button == glfw.MOUSE_BUTTON_RIGHT:   self.inp.right_drag  = pressed
            elif button == glfw.MOUSE_BUTTON_MIDDLE: self.inp.middle_drag = pressed

        def on_scroll(win, xoff, yoff):
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -yoff/self.cfg.camera_zoom_sens, self.scene, self.cam)

        def on_key(win, key, scancode, action, mods):
            pressed = action != glfw.RELEASE
            self.inp.shift = (mods & glfw.MOD_SHIFT) != 0
            if   key == glfw.KEY_LEFT:  self.inp.left  = pressed
            elif key == glfw.KEY_RIGHT: self.inp.right = pressed
            elif key == glfw.KEY_UP:    self.inp.up    = pressed
            elif key == glfw.KEY_DOWN:  self.inp.down  = pressed
            elif key == glfw.KEY_A:     self.inp.z_up   = pressed
            elif key == glfw.KEY_D:     self.inp.z_down = pressed
            elif key == glfw.KEY_F and action == glfw.PRESS: self.inp.grip_close = True
            elif key == glfw.KEY_G and action == glfw.PRESS: self.inp.grip_open  = True
            elif key == glfw.KEY_B and action == glfw.PRESS: self.inp.reset = True
            elif key == glfw.KEY_V and action == glfw.PRESS: self.inp.save_state = True
            # Joint jog bindings: i=0..6 => (1/q), (2/w), (3/e), (4/r), (5/t), (6/y), (7/u)
            # Convention: number increases joint angle, letter decreases
            if   key == glfw.KEY_1: self.inp.jog_plus[0]  = pressed
            elif key == glfw.KEY_Q: self.inp.jog_minus[0] = pressed
            elif key == glfw.KEY_2: self.inp.jog_plus[1]  = pressed
            elif key == glfw.KEY_W: self.inp.jog_minus[1] = pressed
            elif key == glfw.KEY_3: self.inp.jog_plus[2]  = pressed
            elif key == glfw.KEY_E: self.inp.jog_minus[2] = pressed
            elif key == glfw.KEY_4: self.inp.jog_plus[3]  = pressed
            elif key == glfw.KEY_R: self.inp.jog_minus[3] = pressed
            elif key == glfw.KEY_5: self.inp.jog_plus[4]  = pressed
            elif key == glfw.KEY_T: self.inp.jog_minus[4] = pressed
            elif key == glfw.KEY_6: self.inp.jog_plus[5]  = pressed
            elif key == glfw.KEY_Y: self.inp.jog_minus[5] = pressed
            elif key == glfw.KEY_7: self.inp.jog_plus[6]  = pressed
            elif key == glfw.KEY_U: self.inp.jog_minus[6] = pressed

        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_scroll_callback(window, on_scroll)
        glfw.set_key_callback(window, on_key)

        return on_key  # allow further enhancement by recorder controls


# ----------------------------- Robot Access ----------------------------
class RobotModel:
    def __init__(self, cfg: TeleopConfig):
        xml = Path(cfg.xml_path)
        if not xml.exists():
            print(f"[ERROR] XML not found: {xml}")
            sys.exit(1)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg

        self.ee_site_id, self.ee_site_name = self._pick_ee_site_id()
        self.arm_dof = self._dof_indices_for_joints(cfg.arm_joint_names)
        if len(self.arm_dof) != 7:
            raise RuntimeError(f"Expected 7 arm DoFs, got {len(self.arm_dof)}")
        self.arm_act_ids = [self._actuator_index(nm) for nm in cfg.arm_act_names]
        j7_abs = self._dof_indices_for_joints(("joint7",))[0]
        self.pos7 = int(np.where(self.arm_dof == j7_abs)[0][0])

        try:
            self.grip_act = self._actuator_index(cfg.gripper_act_name)
            self.has_gripper = True
            lo, hi = self.model.actuator_ctrlrange[self.grip_act]
            self.grip_open_val = float(hi)
            self.grip_close_val = float(lo)
        except RuntimeError:
            self.has_gripper = False
            self.grip_act = None
            self.grip_open_val = self.grip_close_val = 0.0

        mujoco.mj_forward(self.model, self.data)
        # Build name maps for bodies and joints
        self.body_name_to_id = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(self.model.nbody)}
        self.jnt_name_to_qposadr = {}
        for j in range(self.model.njnt):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm:
                self.jnt_name_to_qposadr[nm] = int(self.model.jnt_qposadr[j])

    # name/id helpers
    def _actuator_index(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' not found.")
        return aid

    def _dof_indices_for_joints(self, joints: Tuple[str, ...]) -> np.ndarray:
        idxs: List[int] = []
        for jn in joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"Joint '{jn}' not found.")
            dofadr = self.model.jnt_dofadr[jid]
            jtype  = self.model.jnt_type[jid]
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                dofnum=1
            elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                dofnum=3
            elif jtype == mujoco.mjtJoint.mjJNT_FREE:
                dofnum=6
            else:
                raise RuntimeError(f"Unsupported joint type for '{jn}' (type={jtype}).")
            idxs.extend(range(dofadr, dofadr + dofnum))
        return np.array(idxs, dtype=int)

    def _pick_ee_site_id(self) -> Tuple[int, str]:
        for nm in self.cfg.ee_site_candidates:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, nm)
            if sid >= 0:
                return sid, nm
        sites = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)]
        raise RuntimeError(f"No EE site found. Tried {self.cfg.ee_site_candidates}. Sites present: {sites}")

    # kinematics
    def ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.copy(self.data.site_xpos[self.ee_site_id])
        R = np.copy(self.data.site_xmat[self.ee_site_id]).reshape(3,3)
        return pos, R

    def jacobian_pos(self) -> np.ndarray:
        nv = self.model.nv
        Jp = np.zeros((3, nv)); Jr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_site_id)
        return Jp[:, self.arm_dof]


# ------------------------------ IK/Control -----------------------------
class IKController:
    def __init__(self, cfg: TeleopConfig, robot: RobotModel):
        self.cfg = cfg
        self.robot = robot

    def compute_qdot(self, des_pos: np.ndarray, des_yaw: float) -> np.ndarray:
        cur_pos, cur_R = self.robot.ee_pose()
        Jp = self.robot.jacobian_pos()
        # Orientation Jacobian (angular velocity)
        nv = self.robot.model.nv
        Jp_full = np.zeros((3, nv)); Jr_full = np.zeros((3, nv))
        mujoco.mj_jacSite(self.robot.model, self.robot.data, Jp_full, Jr_full, self.robot.ee_site_id)
        Jr = Jr_full[:, self.robot.arm_dof]

        e_pos = des_pos - cur_pos
        v_pos = self.cfg.kp_pos * e_pos
        JJt = Jp @ Jp.T
        qdot_pos = Jp.T @ np.linalg.solve(JJt + (self.cfg.damping_lambda**2)*np.eye(3), v_pos)

        q_arm = self.robot.data.qpos[self.robot.arm_dof]
        z = np.zeros(7)
        j_idx_upright = 1
        q_upright = q_arm[j_idx_upright]
        barrier_eps = math.radians(5.0)

        def soft_barrier(q, limit, eps):
            return math.tanh((abs(q) - limit) / eps) * (1.0 if q >= 0.0 else -1.0)

        z[j_idx_upright] += -self.cfg.j1_center_k * q_upright
        z[j_idx_upright] += -self.cfg.j1_limit_k  * soft_barrier(q_upright, self.cfg.j1_limit, barrier_eps)

        j_idx_prefbend = 3
        q_prefbend = q_arm[j_idx_prefbend]
        z[j_idx_prefbend] += -self.cfg.j2_pref_k * (q_prefbend - self.cfg.j2_pref)

        Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + (self.cfg.damping_lambda**2)*np.eye(3), np.eye(3))
        N = np.eye(7) - Jp_pinv @ Jp
        # Secondary orientation task: regulate yaw about world Z
        yaw_cur = math.atan2(cur_R[1,0], cur_R[0,0])
        yaw_err = wrap_pi(yaw_cur - des_yaw)
        w = np.array([0.0, 0.0, -self.cfg.kp_yaw * yaw_err])
        JrJrT = Jr @ Jr.T + (self.cfg.damping_lambda**2) * np.eye(3)
        qdot_yaw = Jr.T @ np.linalg.solve(JrJrT, w)

        qdot = qdot_pos + N @ (z + qdot_yaw)
        return qdot


# ------------------------------ Rendering ------------------------------
class Renderer:
    def __init__(self, cfg: TeleopConfig, robot: RobotModel):
        self.cfg = cfg
        self.robot = robot
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.robot.model, maxgeom=20000)
        # MjrContext requires a current OpenGL context; create after window init
        self.ctx = None
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.0
        self.cam.azimuth = 160.0
        self.cam.elevation = -65.0
        self.cam.lookat = np.array([0.5,0.0,0.35])

    def initialize_gl(self):
        if self.ctx is None:
            self.ctx = mujoco.MjrContext(self.robot.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    @staticmethod
    def draw_target_marker(scene, pos, radius, rgba):
        n = scene.ngeom
        if n >= scene.maxgeom:
            return
        g = scene.geoms[n]
        size = np.array([radius, 0.0, 0.0], dtype=np.float64)
        pos  = np.asarray(pos, dtype=np.float64).reshape(3)
        mat  = np.eye(3, dtype=np.float64).reshape(9)
        rgba = np.asarray(rgba, dtype=np.float32).reshape(4)
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
        g.segid = 1
        scene.ngeom += 1

    def render(self, window, des_pos: np.ndarray):
        # Ensure GL context-dependent resources exist
        if self.ctx is None:
            self.initialize_gl()
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.robot.model, self.robot.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        Renderer.draw_target_marker(self.scene, des_pos, self.cfg.target_radius, np.array(self.cfg.target_rgba))
        mujoco.mjr_render(viewport, self.scene, self.ctx)
        overlay = (
            "XY: ←/→,↑/↓ | Z: A/D | Grip: F/G | "
            "Joint jog (num=+ letter=-): 1/Q, 2/W, 3/E, 4/R, 5/T, 6/Y, 7/U (Shift=coarse) | ESC quits"
        )
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, overlay, "", self.ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()


# ------------------------------ Gripper -------------------------------
class GripperController:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.target = self.robot.grip_open_val

    def update_from_input(self, inp: InputState):
        if not self.robot.has_gripper:
            return
        if inp.grip_close:
            self.target = self.robot.grip_close_val
        if inp.grip_open:
            self.target = self.robot.grip_open_val

    def apply(self):
        if self.robot.has_gripper:
            self.robot.data.ctrl[self.robot.grip_act] = self.target


# ------------------------------ Teleop App ----------------------------
class TeleopApp:
    def __init__(self, cfg: Optional[TeleopConfig] = None):
        self.cfg = cfg or TeleopConfig()
        self.robot = RobotModel(self.cfg)
        self.ik = IKController(self.cfg, self.robot)
        self.renderer = Renderer(self.cfg, self.robot)
        self.inp_ctrl = InputController(self.cfg, self.robot.model, self.renderer.scene, self.renderer.cam)
        self.gripper = GripperController(self.robot)
        # Sticky attach state
        self.attached: Optional[str] = None
        self.attach_offset_pos = np.zeros(3)
        self.attach_offset_mat = np.eye(3)
        # Saved state (memory)
        self.saved_qpos: Optional[np.ndarray] = None
        self.saved_qvel: Optional[np.ndarray] = None

        # Try to auto-load latest saved state from disk
        try:
            from pathlib import Path
            import glob
            save_dir = Path("/home/adam/Documents/coding/autonomous/finetuning/saved_robot_states")
            if save_dir.exists():
                files = sorted(save_dir.glob("state_*.npz"))
                if files:
                    latest = files[-1]
                    arr = np.load(latest)
                    self.saved_qpos = arr["qpos"]; self.saved_qvel = arr["qvel"]
                    # Apply immediately
                    self.robot.data.qpos[:] = self.saved_qpos
                    self.robot.data.qvel[:] = self.saved_qvel
                    mujoco.mj_forward(self.robot.model, self.robot.data)
                    print(f"[INFO] Loaded saved robot state: {latest}")
        except Exception as e:
            print("[WARN] Could not auto-load saved state:", e)

        # desired targets
        ee_pos, ee_R = self.robot.ee_pose()
        self.des_pos = ee_pos.copy()
        # Track desired yaw (about world Z) from current orientation
        self.des_j7 = float(math.atan2(ee_R[1,0], ee_R[0,0]))

    def _create_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(1280, 820, "MuJoCo Panda Teleop (SOLID)", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("GLFW window create failed")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        return window

    def run(self):
        window = self._create_window()

        on_key = self.inp_ctrl.install(window)
        recorder = create_lerobot_recorder(self.robot.model, self.robot.data, "panda_teleop_dataset")
        enhanced_on_key = add_lerobot_controls(recorder, on_key)
        glfw.set_key_callback(window, enhanced_on_key)
        # Initialize GL-dependent renderer state now that a context exists
        self.renderer.initialize_gl()

        dt_ctrl = 1.0 / self.cfg.ctrl_hz
        last_time = time.time()
        accum = 0.0

        print("[INFO] EE site:", self.robot.ee_site_name, "Controls: arrows=XY, A/D=Z, joint jog 1/Q..7/U, F/G=grip, V=save, B=reset, ESC=quit")

        while not glfw.window_should_close(window):
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            accum += dt * self.cfg.ctrl_hz
            substeps = min(self.cfg.subst_steps, int(accum))
            accum -= substeps

            if recorder.is_recording:
                current_action = self.robot.data.ctrl[self.robot.arm_act_ids].copy()
                recorder.record_frame(action=current_action, done=False)

            for _ in range(substeps):
                self._tick_control(dt_ctrl)
                mujoco.mj_step(self.robot.model, self.robot.data)

            self.renderer.render(window, self.des_pos)

        glfw.terminate()
        if recorder.is_recording:
            recorder.stop_recording()
        recorder.finalize_dataset()

    def _tick_control(self, dt_ctrl: float):
        # Save state if requested
        if self.inp_ctrl.inp.save_state:
            self.saved_qpos = np.copy(self.robot.data.qpos)
            self.saved_qvel = np.copy(self.robot.data.qvel)
            # Also write to disk (timestamped in saved_robot_states)
            try:
                import os, time
                os.makedirs("saved_robot_states", exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"saved_robot_states/state_{ts}.npz"
                np.savez(path, qpos=self.saved_qpos, qvel=self.saved_qvel)
                print(f"[INFO] Saved robot state to {path}")
            except Exception as e:
                print("[WARN] Could not save robot state to file:", e)

        # Reset scene if requested
        if self.inp_ctrl.inp.reset:
            if self.saved_qpos is not None and self.saved_qvel is not None:
                self.robot.data.qpos[:] = self.saved_qpos
                self.robot.data.qvel[:] = self.saved_qvel
                mujoco.mj_forward(self.robot.model, self.robot.data)
                print("[INFO] Reset to saved state (V)")
            else:
                mujoco.mj_resetData(self.robot.model, self.robot.data)
                mujoco.mj_forward(self.robot.model, self.robot.data)
                print("[INFO] Reset to XML default state")
            ee_pos, _ = self.robot.ee_pose()
            self.des_pos = ee_pos.copy()
            self.attached = None

        # Cartesian target updates (for IK mode)
        if self.inp_ctrl.inp.left:  self.des_pos[0] -= self.cfg.xy_step
        if self.inp_ctrl.inp.right: self.des_pos[0] += self.cfg.xy_step
        if self.inp_ctrl.inp.up:    self.des_pos[1] += self.cfg.xy_step
        if self.inp_ctrl.inp.down:  self.des_pos[1] -= self.cfg.xy_step
        if self.inp_ctrl.inp.z_up:   self.des_pos[2] += self.cfg.z_step
        if self.inp_ctrl.inp.z_down: self.des_pos[2] -= self.cfg.z_step

        # Joint jog takes precedence over IK when any jog key is pressed
        jog_active = any(self.inp_ctrl.inp.jog_plus) or any(self.inp_ctrl.inp.jog_minus)
        q = self.robot.data.qpos[self.robot.arm_dof]
        if jog_active:
            step = self.cfg.joint_step * (self.cfg.joint_step_coarse_factor if self.inp_ctrl.inp.shift else 1.0)
            delta = np.zeros(7)
            for i in range(7):
                if self.inp_ctrl.inp.jog_plus[i]:
                    delta[i] += step
                if self.inp_ctrl.inp.jog_minus[i]:
                    delta[i] -= step
            # Directly apply joint increments to state (force movement), respecting actuator ctrlrange
            q_des = q + delta
            for i, aid in enumerate(self.robot.arm_act_ids):
                lo, hi = self.robot.model.actuator_ctrlrange[aid]
                q_des[i] = float(clamp(q_des[i], lo, hi))
            # Write new joint positions directly and re-forward kinematics
            self.robot.data.qpos[self.robot.arm_dof] = q_des
            mujoco.mj_forward(self.robot.model, self.robot.data)
        else:
            qdot = self.ik.compute_qdot(self.des_pos, self.des_j7)
            q_des = q + qdot * dt_ctrl

        # Handle stickiness and gripper before clearing one-shot flags
        self._handle_grasp_stickiness()
        self.gripper.update_from_input(self.inp_ctrl.inp)
        self.inp_ctrl.inp.reset_oneshot()

        self.robot.data.ctrl[:] = 0.0
        for i, aid in enumerate(self.robot.arm_act_ids):
            lo, hi = self.robot.model.actuator_ctrlrange[aid]
            self.robot.data.ctrl[aid] = float(clamp(q_des[i], lo, hi))
        # sticky grasp maintenance already updated above before reset
        self.gripper.apply()

    def _handle_grasp_stickiness(self):
        # Attach when closing near a known object
        if self.inp_ctrl.inp.grip_close and self.attached is None:
            ee_pos, ee_R = self.robot.ee_pose()
            for body_name in ("wine_bottle", "cup"):
                bid = self.robot.body_name_to_id.get(body_name, -1)
                if bid < 0:
                    continue
                body_pos = np.copy(self.robot.data.xpos[bid])
                if np.linalg.norm(body_pos - ee_pos) < self.cfg.grasp_capture_dist:
                    # compute offset
                    body_R = self.robot.data.xmat[bid].reshape(3,3)
                    self.attach_offset_mat = ee_R.T @ body_R
                    self.attach_offset_pos = ee_R.T @ (body_pos - ee_pos)
                    self.attached = body_name
                    print(f"[INFO] Attached: {body_name}")
                    break

        # Release on open
        if self.inp_ctrl.inp.grip_open:
            self.attached = None

        # Drive attached body's free joint qpos to follow EE (strong stickiness)
        if self.attached is not None:
            jname = "bottle_free" if self.attached == "wine_bottle" else ("cup_free" if self.attached == "cup" else None)
            if jname and jname in self.robot.jnt_name_to_qposadr:
                adr = self.robot.jnt_name_to_qposadr[jname]
                ee_pos, ee_R = self.robot.ee_pose()
                R = ee_R @ self.attach_offset_mat
                p = ee_pos + ee_R @ self.attach_offset_pos
                # rotation to quaternion wxyz
                tr = R[0,0] + R[1,1] + R[2,2]
                if tr > 0:
                    qw = math.sqrt(1.0 + tr) / 2.0
                    qx = (R[2,1] - R[1,2]) / (4*qw)
                    qy = (R[0,2] - R[2,0]) / (4*qw)
                    qz = (R[1,0] - R[0,1]) / (4*qw)
                else:
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                self.robot.data.qpos[adr:adr+7] = np.array([p[0], p[1], p[2], qw, qx, qy, qz])
                # Zero velocities to keep object glued
                dofadr = self.robot.model.jnt_dofadr[mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, jname)]
                self.robot.data.qvel[dofadr:dofadr+6] = 0.0
                mujoco.mj_forward(self.robot.model, self.robot.data)


def main():
    TeleopApp().run()


if __name__ == "__main__":
    main()
