from physics_state_extractor import PhysicsStateExtractor
from action_manager import ActionManager
from camera_renderer import CameraRenderer
from time import time
class RobotBodyControl:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.physics_extractor = PhysicsStateExtractor(model, data)
        #self.user_request_state = UserRequestState()
        #self.embodiment_manager = EmbodimentManager(model)
        self.action_manager = ActionManager(data)
        self.camera = CameraRenderer(model, data)

    def apply_commands(self,packet):
        # Placeholder logic for one control cycle
        joint_targets = packet.action
        if joint_targets is None:
            raise ValueError("Joint targets must be provided in the packet under 'action' key.")
        robotId= packet.robot_id
        indices = [
        i for i in range(self.model.nu)
        #if semodel.actuator_id2name(i).startswith(robotId)
         ]
        self.action_manager.apply_joint_targets(indices,joint_targets)



    def fill_packet(self, packet):
       print(f"[fill_packet] Called with robot_id = {packet.robot_id}")
       try:
            robot_id = packet.robot_id
            print("[fill_packet] Filling packet with joint state and images")
            joints_dict = self.physics_extractor.get_joint_state(robot_id)
            print("[fill_packet] Got joint state")

            packet.qpos = joints_dict['qpos']
            packet.qvel = joints_dict['qvel']
            packet.joint_names = joints_dict['joint_names']

            # imgs_Dict = {}
            # self.camera.set_camera(packet.robot_id + "_1")
            # imgs_Dict[packet.robot_id + "_1"] = self.camera.get_rgb_image()
            # print("[fill_packet] Got image 1")

            # self.camera.set_camera(packet.robot_id + "_2")
            # imgs_Dict[packet.robot_id + "_2"] = self.camera.get_rgb_image()
            # print("[fill_packet] Got image 2")

            # packet.images = imgs_Dict
            # print("[fill_packet] Packet filled")

            # packet.time= time.time()
            # print("[fill_packet] Packet time set")
       except Exception as e:
            print(f"[fill_packet ERROR] {e}")
            raise
       return packet
