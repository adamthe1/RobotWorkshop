from .physics_state_extractor import PhysicsStateExtractor
from .action_manager import ActionManager
from .camera_renderer import CameraRenderer
from time import time
import mujoco
from logger_config import get_logger

logger = get_logger('RobotBodyControl')
class RobotBodyControl:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.physics_extractor = PhysicsStateExtractor(model, data)
        #self.user_request_state = UserRequestState()
        #self.embodiment_manager = EmbodimentManager(model)
        self.action_manager = ActionManager(data)
        self.camera = CameraRenderer(model, data)

    def apply_commands(self,packet):
        # Placeholder logic for one control cycle
        joint_targets = packet.action
        if joint_targets is None:
            raise ValueError("Joint targets must be provided in the packet under 'action' key.")
        robotId= packet.robot_id
        indices = [
        i for i in range(self.model.nu)
        if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(robotId)
         ]
        self.action_manager.apply_joint_targets(indices,joint_targets)



    def fill_packet(self, packet):
       logger.debug("[fill_packet] Called with robot_id = %s", packet.robot_id)
       try:
            robot_id = packet.robot_id
            logger.debug("[fill_packet] Filling packet with joint state and images")
            joints_dict = self.physics_extractor.get_joint_state(robot_id)
            logger.debug("[fill_packet] Got joint state")

            packet.qpos = joints_dict['qpos']
            packet.qvel = joints_dict['qvel']
            packet.joint_names = joints_dict['joint_names']

            imgs_Dict = {}
            self.camera.set_camera(packet.robot_id + "_cam1")
            imgs_Dict[packet.robot_id + "_cam1"] = self.camera.get_rgb_image()
            logger.debug("[fill_packet] Got image 1")

            # self.camera.set_camera(packet.robot_id + "_cam2")
            # imgs_Dict[packet.robot_id + "_cam2"] = self.camera.get_rgb_image()
            # print("[fill_packet] Got image 2")

            packet.images = imgs_Dict
            logger.debug("[fill_packet] Packet filled")

            # packet.time= time.time()
            logger.debug("[fill_packet] Packet time set")
       except Exception as e:
            logger.debug("[fill_packet ERROR] %s", e)
            raise
       return packet
