import numpy as np
from PIL import Image

from .args import VISION_WEIGHTS_PATH

class DummyVisionModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
    def classify(self, img):
        return "dummy_submission"
    def is_done(self, img, current_submission):
        return False

class MissionStatus:
    def __init__(self):
        self.vision_model = DummyVisionModel(VISION_WEIGHTS_PATH)

    def sub_mission_status(self, robot_status):
        """
        robot_status: dict with keys like
            - 'image': np.ndarray or PIL.Image
            - 'joint_state': dict or np.ndarray
            - 'current_mission': str
            - 'current_submission': str
        Returns:
            - {'subtask': predicted_submission, 'done': bool}
        """
        img = robot_status['image']
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # Predict submission and completion using the vision model
        pred = self.vision_model.classify(img)  # Expects classification method for submission
        done = self.vision_model.is_done(img, robot_status['current_submission'])  # Expects method for completion
        return {'subtask': pred, 'done': done}
