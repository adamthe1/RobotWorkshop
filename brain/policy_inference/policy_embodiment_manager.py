from logger_config import get_logger
from mujoco_folder.robot_control.embodiment_manager import EmbodimentManager as MujocoEmbodimentManager

logger = get_logger("PolicyEmbodimentManager")

class PolicyEmbodimentManager(MujocoEmbodimentManager):
    def __init__(self, robot_dict=None):
        super().__init__(robot_dict=robot_dict)
    pass