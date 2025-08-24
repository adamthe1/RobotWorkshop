# --- action_manager.py ---
class ActionManager:
    def __init__(self, data):
        self.data = data

    def apply_joint_targets(self, joint_targets):
        self.data.ctrl[:len(joint_targets)] = joint_targets

