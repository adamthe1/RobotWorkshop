
from typing import Optional
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Packet:
    robot_id: str
    action: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = None
    qvel: Optional[np.ndarray] = None
    joint_names: Optional[List[str]] = None
    wall_camera: Optional[np.ndarray] = None
    wrist_camera: Optional[np.ndarray] = None
    current_submission: Optional[str] = None
    current_mission: Optional[str] = None
    time: Optional[float] = None

