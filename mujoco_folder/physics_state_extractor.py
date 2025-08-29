import mujoco
import numpy as np  
from logger_config import get_logger
from .embodiment_manager import EmbodimentManager

logger = get_logger("PhysicsStateExtractor")

class PhysicsStateExtractor:
    def __init__(self, model, data):
        self.model = model
        self.data = data  # Keep for legacy compatibility, but prefer snapshots
        self.embodiment_manager = EmbodimentManager(model)

    def get_joint_state(self, robot_id, snapshot=None):
        """Get joint state from snapshot (preferred) or direct data access (legacy)"""
        joint_names = []
        qpos_list = []
        qvel_list = []

        if self.model is None:
            raise ValueError("Model not initialized. Call load_scene() first.")
        if not isinstance(robot_id, str): 
            raise TypeError("Robot ID must be a string.")
            
        # Use snapshot if provided, otherwise fall back to direct data access
        if snapshot is not None:
            qpos_data = snapshot.qpos
            qvel_data = snapshot.qvel
            logger.debug(f"Using snapshot for joint state extraction (version {snapshot.version})")
        else:
            if self.data is None:
                raise ValueError("Data not initialized and no snapshot provided.")
            qpos_data = self.data.qpos
            qvel_data = self.data.qvel
            logger.debug("Using direct data access for joint state (legacy mode)")
            
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

            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            qvel_addr = int(self.model.jnt_dofadr[joint_id])

            qpos_list.append(float(qpos_data[qpos_addr]))
            qvel_list.append(float(qvel_data[qvel_addr]))
            joint_names.append(jname)

        return {
            'qpos': qpos_list,
            'qvel': qvel_list,
            'joint_names': joint_names
        }

    def get_body_pose(self, body_name, snapshot=None):
        """Get body pose from snapshot (preferred) or direct data access (legacy)"""
        body_id = self.model.body(body_name).id
        
        if snapshot is not None:
            return snapshot.xpos[body_id], snapshot.xquat[body_id]
        else:
            if self.data is None:
                raise ValueError("Data not initialized and no snapshot provided.")
            return self.data.xpos[body_id], self.data.xquat[body_id]
