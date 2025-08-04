import numpy as np
from PIL import Image

from dotenv import load_dotenv
import os
from logger_config import get_logger

load_dotenv()

class DummyVisionModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
    def classify(self, img):
        return "dummy_submission"
    def is_done(self, img, current_submission):
        return False

class MissionStatus:
    def __init__(self):

        self.vision_model = None # DummyVisionModel(os.getenv("VISION_MODEL_PATH", 'dummy'))
        self.submission_counter = {}  # this is only for debugging
        self.logger = get_logger('MissionStatus')

    def sub_mission_status(self, packet):
        """
        robot_status: dict with keys like
            - 'image': np.ndarray or PIL.Image
            - 'joint_state': dict or np.ndarray
            - 'current_mission': str
            - 'current_submission': str
        Returns:
            - {'subtask': predicted_submission, 'done': bool}
        """
        if self.vision_model is None:
            if packet.submission in self.submission_counter:
                self.submission_counter[packet.submission] += 1
                if self.submission_counter[packet.submission] > 10:
                    self.logger.debug(f"Submission {packet.submission} has been repeated more than 10 times, go to next.")
                    return {'subtask': packet.submission, 'done': True}
            else:
                self.submission_counter[packet.submission] = 1
            return {'subtask': packet.submission, 'done': False}
        img = packet
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # Predict submission and completion using the vision model
        pred = self.vision_model.classify(img)  # Expects classification method for submission
        done = self.vision_model.is_done(img, robot_status['current_submission'])  # Expects method for completion
        return {'subtask': pred, 'done': done}
