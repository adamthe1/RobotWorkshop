from .missions import SUPPORTED_MISSIONS

import queue
from control_panel.missions import SUPPORTED_MISSIONS
from control_panel.mission_status import MissionStatus
from control_panel.robot_queue_locks import QueueServer, QueueClient
from dotenv import load_dotenv
import os
from logger_config import get_logger

load_dotenv()

class MissionManager:
    def __init__(self):
        """Initialize MissionManager, load status checker, and mission queue."""
        self.logger = get_logger('MissionManager')
        self.queue_client = QueueClient()
        self.status_checker = MissionStatus()
        self.logger.info("Queue server started successfully")
        self.queue_client.connect()
        self.logger.info("Queue client connected successfully")

    def set_robot_list(self, robot_list):
        """Set the list of robots managed by this MissionManager."""
        self.logger.info(f"Setting robot_list to: {robot_list!r}")
        if not isinstance(robot_list, list):
            self.logger.error("robot_list must be a list")
            raise ValueError("robot_list must be a list")
        self.queue_client.add_robot_ids(robot_list)

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
    
    def get_next_mission(self, robot_id):
        """
        If block=True, will block until an item is available
        """
        if self.queue_client.get_robot_lock(robot_id) == 1:
            self.logger.error('Not supposed to be locked')
            return None
        if self.queue_client.see_next_robot() == robot_id:
            # remove mission from queue
            result = self.queue_client.get_robot_mission_pair()
            mission = result['mission']
            robot_id_result = result['robot_id']
            if robot_id_result != robot_id:
                self.logger.error('Got different robot than needed for mission')
                raise
            return mission
        else:
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
            # this means the robot is reset and ready for a new mission
            next_mission = self.get_next_mission()
            if next_mission is None:
                return None # No more missions in queue, robot in standby
            return self.get_next_submission(next_mission, None)
            
        next_sub = self.get_next_submission(robot_status['current_mission'], robot_status['current_submission'])
        if next_sub is None:
            return self.reset_before_new_mission()
        return next_sub
    
    
    def get_robot_from_queue(self):
        """Find a free robot that is not currently processing a mission."""
        return self.queue_client.get_robot_from_queue()


