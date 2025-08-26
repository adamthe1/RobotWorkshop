import mujoco
import numpy as np  
from logger_config import get_logger

logger = get_logger("PhysicsStateExtractor")

class PhysicsStateExtractor:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_joint_state(self,robot_id):
        joint_names = []
        qpos_list = []
        qvel_list = []

        if self.model is None or self.data is None:
            raise ValueError("Model or data not initialized. Call load_scene() first.")
        if not isinstance(robot_id, str): 
            raise TypeError("Robot ID must be a string.")
        # Extract joint names, qpos, and qvel from the model   

        for j in range(self.model.njnt):
            logger.debug("Extracting joint %s", j)
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            logger.debug("Extracting joint %s for robot %s", jname, robot_id)
            jtype = self.model.jnt_type[j]
            logger.debug("Joint type for %s is %s", jname, jtype)
            if jtype == mujoco.mjtJoint.mjJNT_FREE or jtype == mujoco.mjtJoint.mjJNT_BALL:
                logger.debug("Skipping joint %s of type %s", jname, jtype)
                continue
            if jname.startswith(robot_id):
                logger.debug("Processing joint %s for robot %s", jname, robot_id)
                addr = self.model.jnt_qposadr[j] # co responding to 
                qpos_list.append(self.data.qpos[addr])
                qvel_list.append(self.data.qvel[addr])
                joint_names.append(jname)

        return {
            'qpos': qpos_list,
            'qvel': qvel_list,
            'joint_names': joint_names
        }

    def get_body_pose(self, body_name):
        body_id = self.model.body(body_name).id
        return self.data.xpos[body_id], self.data.xquat[body_id]
