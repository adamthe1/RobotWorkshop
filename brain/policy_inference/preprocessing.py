import numpy as np
from typing import List
import torch

class Preprocessing:
    def __init__(self, policies: List):
        """Initialize with a list of policies to apply.
        Will accept a packet and return a tensor ready for the policy.
        """
        self.policies = policies

    def preprocess(self, packet):
        """Preprocess the incoming packet for the policy.
        """
        tensor = self.convert_packet_to_tensor(packet)
        for policy in self.policies:
            tensor = policy.apply(tensor)
        return tensor

    def convert_packet_to_tensor(self, packet):
        """Convert the incoming packet to a tensor.
        packet fields:  qpos: Optional[np.ndarray] = None
                        qvel: Optional[np.ndarray] = None
                        joint_names: Optional[List[str]] = None
                        wall_camera: Optional[np.ndarray] = None
                        wrist_camera: Optional[np.ndarray] = None
        """
        # Implement the conversion logic here
        preprocessed_data = {}
        preprocessed_data['qpos'] = torch.tensor(packet.qpos, dtype=torch.float32) if packet.qpos is not None else torch.zeros(7)
        preprocessed_data['qvel'] = torch.tensor(packet.qvel, dtype=torch.float32) if packet.qvel is not None else torch.zeros(7)
        preprocessed_data['wall_camera'] = self.process_camera_image(packet.wall_camera) if packet.wall_camera is not None else torch.zeros((3, 64, 64))
        preprocessed_data['wrist_camera'] = self.process_camera_image(packet.wrist_camera)
        return preprocessed_data
    
    def preprocess_cameras(self, packet):
        """Preprocess camera images from the packet."""
        processed_cameras = {}
        if packet.wall_camera is not None:
            processed_cameras['wall_camera'] = self.process_camera_image(packet.wall_camera)
        if packet.wrist_camera is not None:
            processed_cameras['wrist_camera'] = self.process_camera_image(packet.wrist_camera)
        return processed_cameras