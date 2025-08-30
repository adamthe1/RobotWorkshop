from logger_config import get_logger

logger = get_logger("EmbodimentManager")

class EmbodimentManager:
    def __init__(self, model, robot_dict=None):
        self.model = model
        
        self.logger = get_logger('EmbodimentManager')
        if robot_dict is None:
            self.logger.warning("No robot_dict provided, defaulting to empty dict")
            self.robot_dict = {"panda1": "FrankaPanda"}
        else:
            self.robot_dict = robot_dict

        # Robot type configurations
        self.robot_joint_mapper = {
            'FrankaPanda': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
            'SO101': ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        }
        
        self.robot_camera_mapper = {
            'FrankaPanda': ['wrist_cam', 'main_cam'],
            'SO101': ['wrist_cam', 'base_cam']
        }
        
        self.robot_actuator_mapper = {
            'FrankaPanda': ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8'],
            'SO101': ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        }
        
        # Gripper joint mappings
        self.robot_gripper_mapper = {
            'FrankaPanda': ['finger_joint1', 'finger_joint2'],
            'SO101': ['gripper']
        }
        
        self.add_prefix_bool = True

    def robot_id2robot_model(self, robot_id):
        # Map robot IDs to their corresponding MuJoCo models
        robot_type = self.robot_dict.get(robot_id, None)
        if robot_type is None:
            self.logger.error(f"Unknown robot ID: {robot_id}")
            return "FrankaPanda"  # Default fallback
        return robot_type

    def get_robot_joint_list(self, robot_id):
        # Get the list of joints for a specific robot
        robot_type = self.robot_id2robot_model(robot_id)

        joints = self.robot_joint_mapper.get(robot_type, [])
        joints = self.add_prefix(robot_id, joints)
        if len(joints) == 0:
            self.logger.error(f"No joints found for robot type: {robot_type}")
        return joints
    
    def get_robot_actuator_list(self, robot_id):
        # Get the list of actuators for a specific robot
        robot_type = self.robot_id2robot_model(robot_id)

        actuators = self.robot_actuator_mapper.get(robot_type, [])
        actuators = self.add_prefix(robot_id, actuators)
        if len(actuators) == 0:
            self.logger.error(f"No actuators found for robot type: {robot_type}")
        return actuators

    def get_robot_camera_list(self, robot_id):
        # Get the list of cameras for a specific robot
        robot_type = self.robot_id2robot_model(robot_id)

        cameras = self.robot_camera_mapper.get(robot_type, [])
        cameras = self.add_prefix(robot_id, cameras)
        if len(cameras) == 0:
            self.logger.error(f"No cameras found for robot type: {robot_type}")
        return cameras
    
    def get_robot_gripper_joints(self, robot_id):
        # Get the list of gripper joints for a specific robot
        robot_type = self.robot_id2robot_model(robot_id)

        gripper_joints = self.robot_gripper_mapper.get(robot_type, [])
        gripper_joints = self.add_prefix(robot_id, gripper_joints)
        if len(gripper_joints) == 0:
            self.logger.warning(f"No gripper joints found for robot type: {robot_type}")
        return gripper_joints
    
    def add_prefix(self, robot_id, name_list):
        if self.add_prefix_bool:
            return [f"{robot_id}_{name}" for name in name_list]
        return name_list

    def to_global_format(self):
        # Translate MuJoCo structure into semantic or planning-compatible format
        pass