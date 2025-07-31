# --- action_manager.py ---
class ActionManager:
    def __init__(self, data):
        self.data = data

    def apply_joint_targets(self,indices, joint_targets):
        if len(joint_targets) != len(indices):
             raise ValueError(f"Action length {len(action)} doesn't match robot actuator count {len(indices)}")
        for i, val in zip(indices, joint_targets):
          data.ctrl[i] = val

