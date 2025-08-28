from .physics_state_extractor import PhysicsStateExtractor
from .action_manager import ActionManager
from .camera_renderer import CameraRenderer
from time import time
from .embodiment_manager import EmbodimentManager
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
        self.camera = None
        self.embodiment_manager = EmbodimentManager(model)

    def _get_camera_renderer(self):
        """Get camera renderer, creating it if needed"""
        if self.camera is None:
            try:
                self.camera = CameraRenderer(self.model, self.data)
            except Exception as e:
                logger.error(f"Failed to create camera renderer: {e}")
                return None
        return self.camera
    
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
        """Fill packet with improved error handling"""
        logger.debug(f"[fill_packet] Called with robot_id = {packet.robot_id}")
        
        try:
            robot_id = packet.robot_id
            
            # Get joint state
            joints_dict = self.physics_extractor.get_joint_state(robot_id)
            packet.qpos = joints_dict['qpos']
            packet.qvel = joints_dict['qvel']
            packet.joint_names = joints_dict['joint_names']
            
            # Get camera images
            imgs_dict = {}
            camera_renderer = self._get_camera_renderer()
            
            if camera_renderer:
                camera_list = self.embodiment_manager.get_robot_camera_list(robot_id)
                logger.debug(f"[fill_packet] Getting images for cameras: {camera_list}")
                
                for camera_name in camera_list:
                    try:
                        if camera_renderer.set_camera(camera_name):
                            img = camera_renderer.get_rgb_image()
                            if img is not None:
                                imgs_dict[camera_name] = img
                                logger.debug(f"[fill_packet] Got image for {camera_name}")
                            else:
                                logger.warning(f"[fill_packet] Failed to get image for {camera_name}")
                        else:
                            logger.warning(f"[fill_packet] Failed to set camera {camera_name}")
                    except Exception as e:
                        logger.error(f"[fill_packet] Error with camera {camera_name}: {e}")
            else:
                logger.warning("[fill_packet] Camera renderer not available")
            
            packet.images = imgs_dict
            logger.debug(f"[fill_packet] Packet filled with {len(imgs_dict)} images")
            
        except Exception as e:
            logger.error(f"[fill_packet] ERROR: {e}")
            # Don't re-raise - return partial packet
            if not hasattr(packet, 'images'):
                packet.images = {}
        
        return packet