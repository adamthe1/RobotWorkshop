from .physics_state_extractor import PhysicsStateExtractor
from .action_manager import ActionManager
from .camera_renderer import CameraRenderer
from time import time
from .embodiment_manager import EmbodimentManager
import mujoco
import numpy as np
import threading
from logger_config import get_logger

logger = get_logger('RobotBodyControl')

class MuJoCoSnapshot:
    """Immutable snapshot of MuJoCo simulation state"""
    def __init__(self, model, data, timestamp=None):
        self.timestamp = timestamp or time()
        self.version = 0
        
        # Copy essential state data
        self.qpos = data.qpos.copy()
        self.qvel = data.qvel.copy() 
        self.ctrl = data.ctrl.copy()
        self.actuator_ctrlrange = model.actuator_ctrlrange.copy()
        self.time = float(data.time)
        
        # Copy contact and force data
        self.ncon = int(data.ncon)
        if self.ncon > 0:
            # Convert contact data to numpy array for safe copying
            self.contact = np.array([data.contact[i] for i in range(self.ncon)])
        else:
            self.contact = np.array([])
            
        # Copy site positions (for cameras/attachments)
        self.site_xpos = data.site_xpos.copy()
        self.site_xmat = data.site_xmat.copy()
        
        # Copy body poses
        self.xpos = data.xpos.copy()
        self.xquat = data.xquat.copy()
        self.xmat = data.xmat.copy()
        
        # Copy sensor data if available
        if model.nsensor > 0:
            self.sensordata = data.sensordata.copy()
        else:
            self.sensordata = np.array([])

class ActionStaging:
    """Thread-safe staging area for actions before they're committed to MuJoCo"""
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._staged_ctrl = np.zeros(model.nu)
        self._has_new_data = False
        self._repeat_last_action = True  # Hold last action for gravity compensation
        
    def stage_action(self, indices, values):
        """Stage action values for specific actuator indices"""
        with self._lock:
            for i, val in zip(indices, values):
                if 0 <= i < len(self._staged_ctrl):
                    # Apply actuator limits
                    lo, hi = self.model.actuator_ctrlrange[i]
                    clamped = max(lo, min(hi, float(val)))
                    self._staged_ctrl[i] = clamped
            self._has_new_data = True
            
    def commit_to_data(self, data):
        """Commit staged actions to MuJoCo data (called only by sim thread)"""
        with self._lock:
            if self._has_new_data or self._repeat_last_action:
                data.ctrl[:] = self._staged_ctrl
                if self._has_new_data:
                    logger.debug(f"Committed new actions: {self._staged_ctrl[:min(14, len(self._staged_ctrl))]}")
                self._has_new_data = False
                return True
            return False
    
    def set_repeat_mode(self, repeat):
        """Enable/disable action repeat mode"""
        with self._lock:
            self._repeat_last_action = repeat
    
    def initialize_from_current_state(self, data):
        """Initialize action cache with current control values from MuJoCo data"""
        with self._lock:
            self._staged_ctrl[:] = data.ctrl[:]
            self._has_new_data = False  # Don't treat this as new data
            logger.debug(f"Initialized action cache with current control values: {self._staged_ctrl[:min(4, len(self._staged_ctrl))]}")

class RobotBodyControl:
    def __init__(self, model, data, robot_dict=None):
        self.model = model
        self.data = data
        
        # Initialize snapshot and staging systems
        self.action_staging = ActionStaging(model)
        self._current_snapshot = None
        self._snapshot_lock = threading.Lock()
        self._snapshot_version = 0
        
        # Legacy components (will be updated to use snapshots)
        self.embodiment_manager = EmbodimentManager(model, robot_dict=robot_dict)
        self.physics_extractor = PhysicsStateExtractor(model, data)
        self.action_manager = ActionManager()

                # Initialize camera renderer (no direct data access)
        self.camera_renderer = CameraRenderer(model)
        

        # Create initial snapshot
        self.update_snapshot()

    def update_snapshot(self):
        """Update the current snapshot (called only by sim thread after mj_step)"""
        with self._snapshot_lock:
            self._current_snapshot = MuJoCoSnapshot(self.model, self.data)
            self._current_snapshot.version = self._snapshot_version
            self._snapshot_version += 1

             # Update camera renderer with new snapshot
            self.camera_renderer.update_snapshot(self._current_snapshot)
    
    def get_snapshot(self):
        """Get the current immutable snapshot (thread-safe for consumers)"""
        with self._snapshot_lock:
            return self._current_snapshot
    
    def commit_staged_actions(self):
        """Commit staged actions to data (called only by sim thread before mj_step)"""
        return self.action_staging.commit_to_data(self.data)
    
    def set_action_repeat_mode(self, repeat=True):
        """Enable/disable action repeat mode for gravity compensation"""
        self.action_staging.set_repeat_mode(repeat)
    
    def initialize_action_cache_from_current_state(self):
        """Initialize action cache with current control values to prevent robots from falling"""
        # First, compute the control values needed to maintain current joint positions
        # For position-controlled actuators, we'll use the current joint positions as targets
        for i in range(self.model.nu):
            # Check actuator type
            actuator_id = i
            joint_id = self.model.actuator_trnid[actuator_id, 0]  # Get the joint this actuator controls
            
            if joint_id >= 0 and joint_id < self.model.njnt:
                # Get current joint position
                qpos_addr = self.model.jnt_qposadr[joint_id]
                current_pos = self.data.qpos[qpos_addr]
                
                # Set control to current position (for position actuators)
                # Apply actuator limits
                lo, hi = self.model.actuator_ctrlrange[i]
                clamped_pos = max(lo, min(hi, current_pos))
                self.data.ctrl[i] = clamped_pos
        
        # Now initialize the action cache with these computed control values
        self.action_staging.initialize_from_current_state(self.data)

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
        """Apply incoming joint targets to staged actions (thread-safe).

        Uses EmbodimentManager to resolve actuator names for the given robot
        and maps the provided action vector to those actuators in order.
        Actions are staged and will be committed by the sim thread.
        """
        joint_targets = packet.action
        if joint_targets is None:
            raise ValueError("Joint targets must be provided in the packet under 'action' key.")

        robot_id = str(packet.robot_id)
        # Retrieve the ordered actuator name list for this robot
        actuator_names = self.embodiment_manager.get_robot_actuator_list(robot_id)
        gripper_name = self.embodiment_manager.get_robot_gripper_joints(robot_id)

        if not actuator_names:
            raise ValueError(f"No actuators found for robot_id={robot_id}")
        
        actuator_indices = []
        resolved_names = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                actuator_indices.append(aid)
                resolved_names.append(name)

        gripper_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_name)
        if gripper_index < 0:
            raise ValueError(f"Gripper actuator not found for robot_id={robot_id}: {gripper_name}")

        snapshot = self.get_snapshot()

        actuator_indices, joint_targets = self.action_manager.prepare_joint_targets(snapshot, actuator_indices, gripper_index, joint_targets)
           # Debug sample
        sample_n = min(14, len(actuator_indices))
        logger.debug(f"Staging {len(actuator_indices)} actuator targets for {robot_id}: "
                     f"{[float(joint_targets[i]) for i in range(sample_n)]} ...")

        # Stage the actions (thread-safe)
        self.action_staging.stage_action(actuator_indices, joint_targets)

        # Fill and return updated packet so caller can see post-apply state
        packet = self.fill_packet(packet)
        return packet



    def fill_packet(self, packet, no_camera=False):
        """Fill packet with improved error handling using snapshot data"""
        logger.debug(f"[fill_packet] Called with robot_id = {packet.robot_id}")
        
        try:
            robot_id = packet.robot_id
            
            # Get current snapshot for thread-safe state access
            snapshot = self.get_snapshot()
            if not snapshot:
                logger.warning("[fill_packet] No snapshot available, using direct data access")
                snapshot = None
            
            # Get joint state from snapshot
            joints = self.embodiment_manager.get_robot_joint_list(robot_id)
            joints_dict = self.physics_extractor.get_joint_state(robot_id, joints, snapshot)
            packet.qpos = joints_dict['qpos']
            packet.qvel = joints_dict['qvel']
            packet.joint_names = joints_dict['joint_names']
            
            # Get camera images (this still needs direct access for rendering)
            # Note: Camera rendering is read-only and uses MuJoCo's render APIs
            # which require direct model/data access. This is safe as long as
            # rendering doesn't modify the simulation state.
            # Get camera images using safe snapshot-based rendering
            if not no_camera:
                imgs_dict = self._get_robot_camera_images(robot_id)
                packet.images = imgs_dict
                logger.debug(f"[fill_packet] Packet filled with {len(imgs_dict)} images")
            else:
                packet.images = {}
            
        except Exception as e:
            logger.error(f"[fill_packet] ERROR: {e}")
            # Don't re-raise - return partial packet
            if not hasattr(packet, 'images'):
                packet.images = {}
        
        return packet
    

    def _get_robot_camera_images(self, robot_id):
        """
        Get camera images for a specific robot using snapshot-based rendering.
        Thread-safe and doesn't touch model/data directly.
        """
        try:
            # Get camera list from embodiment manager
            camera_list = self.embodiment_manager.get_robot_camera_list(robot_id)
            if not camera_list:
                logger.warning(f"No cameras defined for robot {robot_id}")
                return {}
            
            logger.debug(f"Getting images for robot {robot_id} cameras: {camera_list}")
            
            # Validate camera names
            valid_cameras, invalid_cameras = self.camera_renderer.validate_camera_names(camera_list)
            
            if invalid_cameras:
                available = self.camera_renderer.get_available_camera_list()
                logger.warning(f"Invalid cameras for robot {robot_id}: {invalid_cameras} Available cameras: {available}")

            if not valid_cameras:
                logger.warning(f"No valid cameras found for robot {robot_id}")
                return {}
            
            # Get images using snapshot-based rendering
            images = self.camera_renderer.get_camera_images(valid_cameras)
            
            # Filter out None values and log results
            valid_images = {name: img for name, img in images.items() if img is not None}
            failed_cameras = [name for name, img in images.items() if img is None]
            
            if failed_cameras:
                logger.warning(f"Failed to render cameras for robot {robot_id}: {failed_cameras}")
            
            logger.debug(f"Successfully got {len(valid_images)} images for robot {robot_id}")
            return valid_images
            
        except Exception as e:
            logger.error(f"Error getting camera images for robot {robot_id}: {e}")
            return {}

    def cleanup(self):
        """Clean up all resources"""
        try:
            self.camera_renderer.cleanup()
            logger.info("RobotBodyControl cleaned up")
        except Exception as e:
            logger.warning(f"Error during RobotBodyControl cleanup: {e}")
