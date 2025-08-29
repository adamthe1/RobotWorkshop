# --- action_manager.py ---
import mujoco
from logger_config import get_logger

class ActionManager:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.logger = get_logger('ActionManager')

    def apply_joint_targets(self, indices, joint_targets):
        """Apply targets to actuators by index with range clamping.

        - indices: list of actuator indices (int) in model order
        - joint_targets: iterable of target values aligned to indices
        """
        self.logger.debug(f"Applying joint targets: {joint_targets} to indices: {indices}")
        for i, val in zip(indices, joint_targets):
            lo, hi = self.model.actuator_ctrlrange[i]
            # Clamp and set
            v = float(val)
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            self.data.ctrl[i] = v
      
