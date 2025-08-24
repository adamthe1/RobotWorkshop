import mujoco
import numpy as np  

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
            print("DEBUG: Extracting joint", j)
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"DEBUG: Extracting joint {jname} for robot {robot_id}")
            jtype = self.model.jnt_type[j]
            print(f"DEBUG: Joint type for {jname} is {jtype}")
            if jtype == mujoco.mjtJoint.mjJNT_FREE or jtype == mujoco.mjtJoint.mjJNT_BALL:
                print(f"DEBUG: Skipping joint {jname} of type {jtype}") 
                continue
            if jname.startswith(robot_id):
                print(f"DEBUG: Processing joint {jname} for robot {robot_id}")
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

