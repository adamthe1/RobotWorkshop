# --- action_manager.py ---
class ActionManager:
    def __init__(self, data):
        self.data = data

    def apply_joint_targets(self,indices, joint_targets):
        # if len(joint_targets) != len(indices):
        #      raise ValueError(f"Action length {len(indices)} doesn't match robot actuator count {len(indices)}")
        # for i, val in zip(indices, joint_targets):
        #    self.data.ctrl[i] = val
        for i in range(len(indices)):
            self.data.ctrl[i] = joint_targets[i]
        print(f"Applied joint targets: {joint_targets} to indices: {indices}")
