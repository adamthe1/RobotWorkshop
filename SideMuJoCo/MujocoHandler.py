class MuJoCoHandler:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.model = None
        self.data = None
        self.robot_control = None

    def load_scene(self):
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot_control = RobotBodyControl(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data