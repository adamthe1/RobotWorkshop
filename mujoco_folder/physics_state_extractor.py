import mujoco
import numpy as np  
from logger_config import get_logger
from .embodiment_manager import EmbodimentManager

logger = get_logger("PhysicsStateExtractor")

class PhysicsStateExtractor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.embodiment_manager = EmbodimentManager(model)

    def get_joint_state(self,robot_id):
        joint_names = []
        qpos_list = []
        qvel_list = []

        if self.model is None or self.data is None:
            raise ValueError("Model or data not initialized. Call load_scene() first.")
        if not isinstance(robot_id, str): 
            raise TypeError("Robot ID must be a string.")
        # Extract joint names, qpos, and qvel from the model
        logger.debug(f"joint amount {self.model.njnt}")
        joints = self.embodiment_manager.get_robot_joint_list(robot_id)
        for j in joints:
            logger.debug("Extracting joint %s", j)
            jname = str(j)
            logger.debug("Extracting joint %s for robot %s", jname, robot_id)
 
            # get joint id from name
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if joint_id < 0:
                logger.debug("Joint name not found: %s (falling back to j as index)", jname)
                joint_id = int(j)  # fallback if j already was index

            addr = int(self.model.jnt_qposadr[joint_id])
            qpos_list.append(float(self.data.qpos[addr]))
            qvel_list.append(float(self.data.qvel[addr]))
            joint_names.append(jname)

        return {
            'qpos': qpos_list,
            'qvel': qvel_list,
            'joint_names': joint_names
        }

    def get_body_pose(self, body_name):
        body_id = self.model.body(body_name).id
        return self.data.xpos[body_id], self.data.xquat[body_id]
