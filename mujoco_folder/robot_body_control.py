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
        self.action_manager = ActionManager(model, data)
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
    
    def apply_commands(self, packet):
        """Apply incoming joint targets to this robot's actuators.

        Uses EmbodimentManager to resolve actuator names for the given robot
        and maps the provided action vector to those actuators in order.
        """
        joint_targets = packet.action
        if joint_targets is None:
            raise ValueError("Joint targets must be provided in the packet under 'action' key.")

        robot_id = str(packet.robot_id)
        # Retrieve the ordered actuator name list for this robot
        actuator_names = self.embodiment_manager.get_robot_actuator_list(robot_id)
        if not actuator_names:
            raise ValueError(f"No actuators found for robot_id={robot_id}")

        # Map names to actuator indices in the model, preserving order
        indices = []
        resolved_names = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                indices.append(aid)
                resolved_names.append(name)
        if not indices:
            raise ValueError(f"Actuator names not found in model for robot_id={robot_id}: {actuator_names}")

        # Align action length to indices length
        if len(joint_targets) < len(indices):
            # Pad missing targets with current ctrl values for stability
            padded = list(joint_targets) + [float(self.data.ctrl[i]) for i in indices[len(joint_targets):]]
            joint_targets = padded
        elif len(joint_targets) > len(indices):
            # Truncate extras
            joint_targets = joint_targets[:len(indices)]

        # Debug sample
        sample_n = min(4, len(indices))
        logger.debug(f"Applying {len(indices)} actuator targets for {robot_id}: "
                     f"{[(resolved_names[i], float(joint_targets[i])) for i in range(sample_n)]} ...")

        self.action_manager.apply_joint_targets(indices, joint_targets)
        # Fill and return updated packet so caller can see post-apply state
        packet = self.fill_packet(packet)
        return packet



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
