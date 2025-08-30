# --- action_manager.py ---
import mujoco
from logger_config import get_logger
import numpy as np

class ActionManager:
    def __init__(self):
        self.logger = get_logger('ActionManager')

    def prepare_joint_targets(self, snapshot, actuator_indices, gripper_index, joint_targets):
        # Map names to actuator indices in the model, preserving order
        
        # Align action length to indices length
        if len(joint_targets) < len(actuator_indices):
            # Get current control values from snapshot for padding
            
            if snapshot:
                padded = list(joint_targets) + [float(snapshot.ctrl[i]) for i in actuator_indices[len(joint_targets):]]
            else:
                # Fallback to zeros if no snapshot available
                padded = list(joint_targets) + [0.0] * (len(actuator_indices) - len(joint_targets))
            joint_targets = padded
        elif len(joint_targets) > len(actuator_indices):
            # Truncate extras
            joint_targets = joint_targets[:len(actuator_indices)]

        if snapshot:
            # clamp joint targets to actuator limits
            for i in range(len(joint_targets)):
                lo, hi = snapshot.actuator_ctrlrange[i]
                joint_targets[i] = self.clamp(joint_targets[i], lo, hi)

        return actuator_indices, joint_targets

    def clamp(self, x, lo, hi):
        return np.minimum(np.maximum(x, lo), hi)
    