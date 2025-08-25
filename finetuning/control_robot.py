#!/usr/bin/env python3
# MuJoCo 3.3.4 – Panda teleop: keyboard target + wrist-only (j6,j7) upright + yaw in nullspace
# Target: ←/→ = X-/X+ | ↑/↓ = Y+/Y- | A/D = Z+/Z-
# Yaw: Q/E (hold Shift for coarse)
# Gripper: F close, G open
# Camera: Right-drag orbit, Middle-drag pan, Scroll zoom

import sys, time, math
from pathlib import Path
import numpy as np
import mujoco
import glfw
from recording import create_lerobot_recorder, add_lerobot_controls

# ============== User config ==============
XML_PATH = "/home/adam/Documents/coding/autonomous/franka_emika_panda/panda_bar_scene_single.xml"
EE_SITE_CANDIDATES = ["ee_site"]
ARM_JOINT_NAMES  = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
ARM_ACT_NAMES    = ["actuator1","actuator2","actuator3","actuator4","actuator5","actuator6","actuator7"]
GRIPPER_ACT_NAME = "actuator8"

# Which local axis of the EE site should be aligned with world +Z to be "parallel to floor"?
UP_AXIS = 2  # 0=X, 1=Y, 2=Z (change if your site frame differs)

# Target step per control tick (while key held)
XY_STEP = 0.01
Z_STEP  = 0.01

# IK gains
Kp_pos          = 12.0         # position gain
Kp_upright      = 8.0          # roll/pitch leveling via (u x z)
Kp_yaw          = 6.0          # yaw (about world z) toward user-set target
DAMPING_LAMBDA  = 0.10         # DLS lambda (same for all pseudo-inverses)

# Control loop
CTRL_HZ = 60.0
DT_CTRL = 1.0 / CTRL_HZ
SUBSTEPS_PER_RENDER = 2

# Marker
TARGET_RADIUS = 0.01
TARGET_RGBA   = np.array([1.0, 0.1, 0.1, 1.0], dtype=float)

# Camera sensitivity (larger = gentler)
CAMERA_ORBIT_SENS = 100.0
CAMERA_PAN_SENS   = 100.0
CAMERA_ZOOM_SENS  = 60.0

# --- Posture/limits (nullspace) ---
J1_LIMIT = np.deg2rad(10.0)   # ±30 deg soft limit for joint 1
J1_LIMIT_K = 4.0              # how strongly to push back when beyond limit
J1_CENTER_K = 200.0  # increase if you want stronger centering
J2_PREF = -0.5               # radians; "bend a little more" (tweak: -0.35 .. -0.8)
J2_PREF_K = 0.8               # pull strength toward the preferred bend
# =========================================

def clamp(x, lo, hi): return np.minimum(np.maximum(x, lo), hi)

def yaw_from_R(R):
    # ZYX convention: yaw = atan2(r10, r00) (robust even if slightly tilted)
    return math.atan2(R[1,0], R[0,0])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def get_site_pose(model, data, site_id):
    pos = np.copy(data.site_xpos[site_id])
    R = np.copy(data.site_xmat[site_id]).reshape(3,3)
    return pos, R

def jacobian_site(model, data, site_id):
    nv = model.nv
    Jp = np.zeros((3, nv)); Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
    return Jp, Jr

def dof_indices_for_joints(model, joint_names):
    idxs = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0: raise RuntimeError(f"Joint '{jn}' not found.")
        dofadr = model.jnt_dofadr[jid]
        jtype  = model.jnt_type[jid]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE): dofnum=1
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:  dofnum=3
        elif jtype == mujoco.mjtJoint.mjJNT_FREE:  dofnum=6
        else: raise RuntimeError(f"Unsupported joint type for '{jn}' (type={jtype}).")
        idxs.extend(range(dofadr, dofadr + dofnum))
    return np.array(idxs, dtype=int)

def actuator_index(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0: raise RuntimeError(f"Actuator '{name}' not found.")
    return aid

def pick_ee_site_id(model):
    for nm in EE_SITE_CANDIDATES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid >= 0: return sid, nm
    sites = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    raise RuntimeError(f"No EE site found. Tried {EE_SITE_CANDIDATES}. Sites present: {sites}")

def draw_target_marker(scene, pos, radius, rgba):
    n = scene.ngeom
    if n >= scene.maxgeom: return
    g = scene.geoms[n]
    size = np.array([radius, 0.0, 0.0], dtype=np.float64)
    pos  = np.asarray(pos, dtype=np.float64).reshape(3)
    mat  = np.eye(3, dtype=np.float64).reshape(9)
    rgba = np.asarray(rgba, dtype=np.float32).reshape(4)
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    g.segid = 1
    scene.ngeom += 1

class InputState:
    def __init__(self):
        # movement
        self.left=False; self.right=False; self.up=False; self.down=False
        self.z_up=False; self.z_down=False
        # yaw
        self.yaw_left=False; self.yaw_right=False; self.shift=False
        # camera
        self.right_drag=False; self.middle_drag=False
        self.last_x=0.0; self.last_y=0.0
        # gripper toggles (one-shot)
        self.grip_close=False; self.grip_open=False
    def reset_oneshot(self):
        self.grip_close=False; self.grip_open=False
def main():
    # ----- Load model -----
    xml = Path(XML_PATH)
    if not xml.exists():
        print(f"[ERROR] XML not found: {xml}")
        sys.exit(1)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data  = mujoco.MjData(model)

    ee_site_id, ee_site_name = pick_ee_site_id(model)
    arm_dof = dof_indices_for_joints(model, ARM_JOINT_NAMES)  # 7 hinge DoFs
    if len(arm_dof) != 7:
        raise RuntimeError(f"Expected 7 arm DoFs, got {len(arm_dof)}")
    arm_act_ids = [actuator_index(model, nm) for nm in ARM_ACT_NAMES]

    # Map joint7 to its index inside the 7-DoF arm subspace
    j7_abs = dof_indices_for_joints(model, ["joint7"])[0]
    pos7 = int(np.where(arm_dof == j7_abs)[0][0])

    # Gripper actuator
    try:
        grip_act = actuator_index(model, GRIPPER_ACT_NAME)
        has_gripper = True
        lo, hi = model.actuator_ctrlrange[grip_act]
        GRIP_OPEN_VAL  = float(hi)
        GRIP_CLOSE_VAL = float(lo)
        grip_target = GRIP_OPEN_VAL
    except RuntimeError:
        print(f"[WARN] No gripper actuator '{GRIPPER_ACT_NAME}'.")
        has_gripper=False; grip_act=None
        GRIP_OPEN_VAL=GRIP_CLOSE_VAL=0.0; grip_target=0.0

    # ----- GLFW window -----
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(1280, 820, "MuJoCo 3.3.4 Panda Teleop (Keyboard + Joint7 Yaw)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window create failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ----- Render objects -----
    cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=20000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.0; cam.azimuth = 160.0; cam.elevation = -65.0; cam.lookat = np.array([0.5,0.0,0.35])

    # ----- Input handlers -----
    inp = InputState()

    def on_cursor(win, x, y):
        dx = x - inp.last_x; dy = y - inp.last_y
        if inp.right_drag:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx/CAMERA_ORBIT_SENS, 0, scene, cam)
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, -dy/CAMERA_ORBIT_SENS, scene, cam)
        elif inp.middle_drag:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_H,   dx/CAMERA_PAN_SENS, 0, scene, cam)
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V,   0,  -dy/CAMERA_PAN_SENS, scene, cam)
        inp.last_x, inp.last_y = x, y

    def on_button(win, button, action, mods):
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_RIGHT:   inp.right_drag  = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE: inp.middle_drag = pressed

    def on_scroll(win, xoff, yoff):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -yoff/CAMERA_ZOOM_SENS, scene, cam)

    def on_key(win, key, scancode, action, mods):
        pressed = action != glfw.RELEASE
        inp.shift = (mods & glfw.MOD_SHIFT) != 0
        if   key == glfw.KEY_LEFT:  inp.left  = pressed
        elif key == glfw.KEY_RIGHT: inp.right = pressed
        elif key == glfw.KEY_UP:    inp.up    = pressed
        elif key == glfw.KEY_DOWN:  inp.down  = pressed
        elif key == glfw.KEY_A:     inp.z_up   = pressed
        elif key == glfw.KEY_D:     inp.z_down = pressed
        elif key == glfw.KEY_Q:     inp.yaw_left  = pressed   # will steer joint7 target
        elif key == glfw.KEY_E:     inp.yaw_right = pressed   # will steer joint7 target
        elif key == glfw.KEY_F and action == glfw.PRESS: inp.grip_close = True
        elif key == glfw.KEY_G and action == glfw.PRESS: inp.grip_open  = True

    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    mujoco.mj_forward(model, data)

    # ----- Desired target init -----
    ee_pos, _ee_R = get_site_pose(model, data, ee_site_id)
    des_pos = ee_pos.copy()

    # Desired heading for joint7: start at current q7
    des_j7 = float(data.qpos[arm_dof][pos7])

    print(f"[INFO] EE site: '{ee_site_name}'. Arrows: XY, A/D: Z±, Q/E: joint7 yaw (Shift=coarse), F/G: grip, ESC: quit.")

    last_time = time.time(); ctrl_accum = 0.0

    # Create LeRobot recorder
    recorder = create_lerobot_recorder(model, data, "panda_teleop_dataset")
    # Enhance key callback with recording controls
    original_on_key = on_key
    enhanced_on_key = add_lerobot_controls(recorder, original_on_key)
    glfw.set_key_callback(window, enhanced_on_key)

    while not glfw.window_should_close(window):
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        now = time.time()
        dt = now - last_time
        last_time = now
        ctrl_accum += dt * CTRL_HZ
        substeps = min(SUBSTEPS_PER_RENDER, int(ctrl_accum))
        ctrl_accum -= substeps

        # Record frame if recording
        if recorder.is_recording:
            current_action = data.ctrl[arm_act_ids].copy()
            recorder.record_frame(action=current_action, done=False)

        for _ in range(substeps):
            # ---- 1) update target position and joint7 target yaw from keys ----
            if inp.left:  des_pos[0] -= XY_STEP
            if inp.right: des_pos[0] += XY_STEP
            if inp.up:    des_pos[1] += XY_STEP
            if inp.down:  des_pos[1] -= XY_STEP
            if inp.z_up:   des_pos[2] += Z_STEP
            if inp.z_down: des_pos[2] -= Z_STEP

            yaw_step = 0.01 if not inp.shift else 0.04
            if inp.yaw_left:  des_j7 += yaw_step
            if inp.yaw_right: des_j7 -= yaw_step

            # ---- 2) Kinematics ----
            cur_pos, _cur_R = get_site_pose(model, data, ee_site_id)
            Jp_full, _Jr_full = jacobian_site(model, data, ee_site_id)
            Jp = Jp_full[:, arm_dof]    # (3,7)

            # ---- 3) Primary: position-only DLS ----
            e_pos = des_pos - cur_pos
            v_pos = Kp_pos * e_pos
            JJt = Jp @ Jp.T
            qdot_pos = Jp.T @ np.linalg.solve(JJt + (DAMPING_LAMBDA**2)*np.eye(3), v_pos)  # (7,)

            # ---- 4) Posture biases in the nullspace ----
            q_arm = data.qpos[arm_dof]  # (7,)

            # (a) Joint index 1: keep near 0 and softly repel near/over ±J1_LIMIT
            j_idx_upright = 1
            q_upright = q_arm[j_idx_upright]
            J1_CENTER_K = 2.0
            BARRIER_EPS = np.deg2rad(5.0)
            def soft_barrier(q, limit, eps):
                return math.tanh((abs(q) - limit) / eps) * (1.0 if q >= 0.0 else -1.0)

            # (b) Joint index 3: prefer specific bend angle J2_PREF
            j_idx_prefbend = 3
            q_prefbend = q_arm[j_idx_prefbend]

            # Build posture bias vector
            z = np.zeros(7)
            # (a) center + soft-limit for joint index 1
            z[j_idx_upright] += -J1_CENTER_K * q_upright
            z[j_idx_upright] += -J1_LIMIT_K  * soft_barrier(q_upright, J1_LIMIT, BARRIER_EPS)
            # (b) preferred bend for joint index 3
            z[j_idx_prefbend] += -J2_PREF_K * (q_prefbend - J2_PREF)

            # ---- 5) Nullspace projector of the position task ----
            Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + (DAMPING_LAMBDA**2)*np.eye(3), np.eye(3))
            N = np.eye(7) - Jp_pinv @ Jp

            # Combine: primary position + nullspace posture
            qdot_7 = qdot_pos + N @ z

            # ---- 6) Joint7 heading-hold (direct term on j7 only) ----
            Kp_j7 = 6.0  # tune 3..10
            q7 = q_arm[pos7]
            q7_err = wrap_pi(q7 - des_j7)  # wrap to [-pi, pi]
            qdot_7[pos7] += -Kp_j7 * q7_err
            # optional small clamp for stability
            qdot_7[pos7] = float(clamp(qdot_7[pos7], -1.0, 1.0))

            # ---- 7) Integrate and command position actuators ----
            q = data.qpos[arm_dof]
            q_des = q + qdot_7 * DT_CTRL

            # Gripper toggles (binary)
            if has_gripper:
                if inp.grip_close: grip_target = GRIP_CLOSE_VAL
                if inp.grip_open:  grip_target = GRIP_OPEN_VAL
            inp.reset_oneshot()

            data.ctrl[:] = 0.0
            for i, aid in enumerate(arm_act_ids):
                lo, hi = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = float(clamp(q_des[i], lo, hi))
            if has_gripper:
                data.ctrl[grip_act] = grip_target

            mujoco.mj_step(model, data)

        # ---- Render ----
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        draw_target_marker(scene, des_pos, TARGET_RADIUS, TARGET_RGBA)
        mujoco.mjr_render(viewport, scene, ctx)

        overlay = "XY: ←/→,↑/↓ | Z: A/D | Joint7 yaw: Q/E (Shift=coarse) | Grip: F/G | ESC quits"
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                           mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, overlay, "", ctx)
        glfw.swap_buffers(window); glfw.poll_events()

    glfw.terminate()

    if recorder.is_recording:
        recorder.stop_recording()
    recorder.finalize_dataset()


if __name__ == "__main__":
    main()
