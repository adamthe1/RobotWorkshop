from logger_config import get_logger

logger = get_logger("EmbodimentManager")

class EmbodimentManager:
    def __init__(self, model):
        self.model = model
        self.robot_type_mapper = {'r1': 'FrankaPanda', 'r2': 'FrankaPanda'}
        self.robot_joint_mapper = {
            'FrankaPanda': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
            }
        self.robot_actuator_mapper = {
            'FrankaPanda': ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8'],
            }
    def robot_id2robot_model(self, robot_id):
        # Map robot IDs to their corresponding MuJoCo models
        robot_type = self.robot_type_mapper.get(robot_id, None)
        if robot_type is None:
            logger.error(f"Unknown robot ID: {robot_id}")
        return robot_type

    def get_robot_joint_list(self, robot_id):
        # Get the list of joints for a specific robot
        robot_type = self.robot_id2robot_model(robot_id)

        joints = self.robot_joint_mapper.get(robot_type, [])
        if len(joints) == 0:
            logger.error(f"No joints found for robot type: {robot_type}")
        return joints

    def to_global_format(self):
        # Translate MuJoCo structure into semantic or planning-compatible format
        pass