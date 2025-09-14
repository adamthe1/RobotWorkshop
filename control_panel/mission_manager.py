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

    @staticmethod
    def set_robot_list(robot_list):
        """Set the list of robots managed by this MissionManager."""
        if not isinstance(robot_list, list):
            raise ValueError("robot_list must be a list")
        with QueueClient() as connection:
            connection.add_robot_ids(robot_list)

    def get_supported_missions(self):
        """Return a numbered string of supported missions."""
        missions = list(SUPPORTED_MISSIONS.keys())
        return "\n".join([f"{i+1}. {m}" for i, m in enumerate(missions)])

    def get_submissions(self, mission):
        """Return list of submissions for the given mission."""
        return SUPPORTED_MISSIONS.get(mission, [])

    def get_next_submission(self, packet):
        """Return next submission for mission"""
        submissions = SUPPORTED_MISSIONS.get(packet.mission, [])
        if packet.submission is None:
            if not submissions:
                self.logger.error(f"No submissions found for mission {packet.mission}")
                raise ValueError("No submissions available for this mission")
            packet.submission = submissions[0]
            return packet

        idx = submissions.index(packet.submission)
        if idx + 1 < len(submissions):
            packet.submission = submissions[idx + 1]
            return packet
        else:
            self.logger.error(f"idx for mission {packet.mission} is out of range")
            raise ValueError("No more submissions available for this mission")


    def is_last_submission(self, packet):
        """Return command to reset before starting a new mission."""
        return packet.submission == SUPPORTED_MISSIONS.get(packet.mission, [])[-1]
    
    def get_next_mission(self, robot_id):
        """
        If block=True, will block until an item is available
        """
        if self.queue_client.get_robot_lock(robot_id) == 1:
            self.logger.error('Not supposed to be locked')
            return {'robot_id': None, 'mission': None}
        if self.queue_client.see_next_robot() == robot_id:
            # remove mission from queue
            
            return self.queue_client.get_robot_mission_pair()
        else:
            return {'robot_id': None, 'mission': None}

    def manage_mission(self, packet):
        """Manage mission based on robot status and return next submission or reset command."""
        if not packet or packet.mission is None:
            raise ValueError("Invalid packet provided")
        
        packet.mission_status = 'ongoing'
        
        if packet.submission is None:
            # added support for completion with episode mapper
            packet = self.get_next_submission(packet)

            self.logger.debug(f"adding first submission to packet {packet.submission} for mission {packet.mission}")
            return packet
        self.logger.debug(f"Checking status for submission {packet.submission} status {packet.submission_status}")
        result = self.status_checker.sub_mission_status(packet)

        packet.submission_status = "completed" if result['done'] else "ongoing"

        if result['done']:
            self.logger.debug(f"Submission {packet.submission} completed for mission {packet.mission}")
            if self.is_last_submission(packet):
                packet.mission_status = 'completed'
                return self.reset_packet(packet)
            else:
                packet = self.get_next_submission(packet)
                packet.submission_status = None
                return packet
            
        return packet
    
    def reset_packet(self, packet):
        self.logger.info(f"Mission {packet.mission} completed for robot {packet.robot_id}")
        packet.mission = None
        packet.mission_status = None
        packet.submission = None
        packet.submission_status = None
        self.put_robot_in_queue(packet.robot_id)
        self.status_checker.submission_counter.clear()  # Reset counter after mission completion
        return packet
    
    def put_robot_in_queue(self, robot_id):
        """Put robot back in queue after mission completion."""
        self.logger.info(f"Putting robot {robot_id} back in queue")
        self.queue_client.enqueue_robot(robot_id)
        return

    def reset_robot_and_mission(self, robot_id, mission):
        """Unlock the robot for new missions."""
        self.logger.info(f"Unlocking robot {robot_id}")
        self.put_robot_in_queue(robot_id)
        self.queue_client.enqueue_mission(mission)

    def get_robot_from_queue(self):
        """Find a free robot that is not currently processing a mission."""
        return self.queue_client.get_robot_from_queue()