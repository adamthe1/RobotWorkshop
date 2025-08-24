# --- action_manager.py ---
class ActionManager:
    def __init__(self, data):
        self.data = data

    def apply_joint_targets(self,indices, joint_targets):
        for i, val in zip(indices, joint_targets):
           self.data.ctrl[i] = val
      
