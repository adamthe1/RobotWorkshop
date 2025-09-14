#!/usr/bin/env python3
"""
Reset the beer glass and bottles' free joints to their original per-robot
indexed poses using compose_scene's layout logic.

- Applies compose-style y offset with spacing 2.5.
- Resets: beer glass, green bottle, yellow bottle.
"""
from typing import Tuple
import os
import numpy as np
import mujoco


def _pos_with_offset(x: float, y: float, z: float, *, y_offset: float = 0.0, robot_index: int = 0, robot_spacing: float = 2.5) -> Tuple[float, float, float]:
    total_y = y + y_offset + (robot_index * robot_spacing)
    return float(x), float(total_y), float(z)


def _freejoint_qpos_addr(model, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid])


def reset_beer_glass_to_default(model, data, robot_id: str, *, x: float = 0.13, y: float = -0.5, z: float = 0.95, y_offset: float = None) -> None:
    """Reset beer glass for given robot_id to default compose position."""
    if y_offset is None:
        try:
            y_offset = float(os.getenv("SCENE_Y_OFFSET", "0.0"))
        except Exception:
            y_offset = 0.0

    # Derive composer index: suffix is typically '0' for first robot → glass index 1
    try:
        suffix = robot_id.split('_')[-1]
        glass_index = int(suffix) + 1
    except Exception as e:
        raise ValueError(f"Unable to parse robot index from robot_id='{robot_id}': {e}")

    # Compute target position using the same layout logic
    pos = _pos_with_offset(x, y, z, y_offset=y_offset, robot_index=glass_index - 1, robot_spacing=2.5)

    joint_name = f"beer_glass_free{glass_index}"
    adr = _freejoint_qpos_addr(model, joint_name)

    # Set [x, y, z, qw, qx, qy, qz] with identity quaternion
    data.qpos[adr:adr+3] = np.asarray(pos, dtype=float)
    data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    mujoco.mj_forward(model, data)


def reset_cup_and_bottles_to_default(model, data, robot_id: str, *, y_offset: float = None) -> None:
    """
    Reset the beer glass and both bottles for the robot identified by robot_id.

    Positions derived from compose_scene.get_scene1:
    - beer_glass_pos = (0.13, -0.5, 0.95)
    - green_bottle_pos = (0.05, 0.5, 0.95)
    - yellow_bottle_pos = (0.2, 0.5, 0.95)
    """
    if y_offset is None:
        try:
            y_offset = float(os.getenv("SCENE_Y_OFFSET", "0.0"))
        except Exception:
            y_offset = 0.0

    try:
        suffix = robot_id.split('_')[-1]
        index = int(suffix) + 1
    except Exception as e:
        raise ValueError(f"Unable to parse robot index from robot_id='{robot_id}': {e}")

    items = [
        (f"beer_glass_free{index}", (0.13, -0.5, 0.95)),
        (f"green_bottle_free{index}", (0.05, 0.5, 0.95)),
        (f"yellow_bottle_free{index}", (0.2, 0.5, 0.95)),
    ]

    for joint_name, (x, y, z) in items:
        pos = _pos_with_offset(x, y, z, y_offset=y_offset, robot_index=index - 1, robot_spacing=2.5)
        adr = _freejoint_qpos_addr(model, joint_name)
        data.qpos[adr:adr+3] = np.asarray(pos, dtype=float)
        data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    mujoco.mj_forward(model, data)
