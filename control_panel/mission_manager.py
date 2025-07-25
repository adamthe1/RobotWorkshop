from .missions import SUPPORTED_MISSIONS

import queue
from .missions import SUPPORTED_MISSIONS
from .mission_status import MissionStatus
from .args import VISION_WEIGHTS_PATH


class MissionManager:
    def __init__(self):
        """Initialize MissionManager, load status checker, and mission queue."""
        self.mission_queue = queue.Queue()
        self.status_checker = MissionStatus()

        self.status_checker = MissionStatus()

    def get_supported_missions(self):
        """Return a numbered string of supported missions."""
        missions = list(SUPPORTED_MISSIONS.keys())
        return "\n".join([f"{i+1}. {m}" for i, m in enumerate(missions)])

    def get_submissions(self, mission):
        """Return list of submissions for the given mission."""
        return SUPPORTED_MISSIONS.get(mission, [])

    def get_next_submission(self, mission, current_submission):
        """Return next submission for mission or None if finished."""
        submissions = SUPPORTED_MISSIONS.get(mission, [])
        if current_submission is None:
            return submissions[0] if submissions else None
        idx = submissions.index(current_submission)
        if idx + 1 < len(submissions):
            return submissions[idx + 1]
        return None 

    def reset_before_new_mission(self):
        """Return command to reset before starting a new mission."""
        return "reset before new mission"
    
    def add_mission_to_queue(self, mission):
        """Add mission name to the mission queue."""
        self.mission_queue.put(mission)

    def get_next_mission(self):
        """Return (without removing) the next mission from the queue or None."""
        if not self.mission_queue.empty():
            return self.mission_queue.queue
        return None

    def manage_mission(self, robot_status):
        """Manage mission based on robot status and return next submission or reset command."""
        if not robot_status or 'current_mission' not in robot_status:
            raise ValueError("Invalid robot status provided")
        if not self.vision_model:
            raise ValueError("No vision model set for status checking")
        
        result = self.status_checker.sub_mission_status(robot_status)

        if not result['done']:
            return robot_status['current_submission']
        
        if result['done'] == 'reset':
            # this means the robot is reset and reaqy for a new mission
            next_mission = self.get_next_mission()
            if next_mission is None:
                return None # No more missions in queue, robot in standby
            return self.get_next_submission(next_mission, None)
            
        next_sub = self.get_next_submission(robot_status['current_mission'], robot_status['current_submission'])
        if next_sub is None:
            return self.reset_before_new_mission()
        return next_sub
