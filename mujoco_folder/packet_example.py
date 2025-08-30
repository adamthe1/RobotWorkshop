
from typing import Optional
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Packet:
    robot_id: int
    action: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = None
    qvel: Optional[np.ndarray] = None
    joint_names: Optional[List[str]] = None
    wall_camera: Optional[np.ndarray] = None
    wrist_camera: Optional[np.ndarray] = None
    submission: Optional[str] = None
    mission: Optional[str] = None
    time: Optional[float] = None

@dataclass
class RobotListPacket:
    robot_id: str = 'robot_list'
    robot_list: List[str] = None
    robot_dict: Optional[dict] = None