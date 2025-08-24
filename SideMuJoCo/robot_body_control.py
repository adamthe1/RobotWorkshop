class RobotBodyControl:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.physics_extractor = PhysicsStateExtractor(model, data)
        self.user_request_state = UserRequestState()
        self.embodiment_manager = EmbodimentManager(model)
        self.action_manager = ActionManager(data)

    def update(self):
        # Placeholder logic for one control cycle
        joint_targets = self.user_request_state.receive()
        self.action_manager.apply_joint_targets(joint_targets)
