from .episode_action_mapper import EpisodeActionMapper
from .joint_test_mapper import JointTestMapper
from .policy_embodiment_manager import PolicyEmbodimentManager
from logger_config import get_logger
from typing import List, Dict, Any
import numpy as np
import time


class PolicyManager:
    def __init__(self, robot_dict: Dict[str, str] = None):
        self.logger = get_logger('PolicyManager')
        robot_types = set(robot_dict.values()) if robot_dict else set()
        self.logger.info(f"Initializing PolicyManager with robot types: {robot_types}")
        self.type_to_policy = {"FrankaPanda": "Episode", "SO101": "Episode"}
        self.policies = {"Episode": EpisodeActionMapper(), "JointTest": JointTestMapper()}
        self.load_types_into_policy(robot_types)

    def get_policy(self, name):
        return self.policies.get(name, None)

    def get_policy_for_type(self, robot_type):
        policy_name = self.type_to_policy.get(robot_type, None)
        if policy_name is None:
            self.logger.warning(f"No policy found for robot type: {robot_type}")
        return policy_name, self.get_policy(policy_name)

    def get_policy_for_robot_id(self, robot_id):
        robot_type = self.robot_dict.get(robot_id, None)
        if robot_type is None:
            self.logger.error(f"No policy available for robot ID: {robot_id}")
            raise ValueError(f"No policy available for robot ID: {robot_id}")
        return self.get_policy_for_type(robot_type)

    def load_types_into_policy(self, types):
        for robot_type in types:
            policy = self.get_policy_for_type(robot_type)
            if policy:
                result = policy.load_type(robot_type)
                if result != True:
                    self.logger.error(f"Failed to load policy for robot type: {robot_type}")
                    raise ValueError(f"Failed to load policy for robot type: {robot_type}")
                
                self.logger.info(f"Loaded policy {policy.__class__.__name__} for robot type: {robot_type}")
    

class PolicyInference:
    def __init__(self, robot_dict):
        self.robot_dict = robot_dict

        self.embodiment_manager = PolicyEmbodimentManager(robot_dict=robot_dict)
        self.policy_manager = PolicyManager(robot_dict=robot_dict)
        self.policy_manager.load_types_into_policy(set(robot_dict.values()))


    def fill_action(self, packet):
        time_now = time.time()
        robot_id = packet.robot_id
        mission = packet.mission

        try:
            # Implement action inference logic based on the observation
            policy_name, policy = self.policy_manager.get_policy_for_robot_id(packet.robot_id)

            if policy_name == 'Episode':
                action = policy.next_action()
                progress = self.mapper.get_progress()
                #self.logger.debug(f"REPLAY: Robot {robot_id} progress: {progress*100:.1f}%")
                if progress == 1.0:
                    packet.mission_status = 'completed'
                    self.logger.info(f"REPLAY: Robot {robot_id} mission completed")
            
        except Exception as e:
                self.logger.error(f"Mapper failed, falling back to dummy: {e}")
                action = None
        else:
            action = None

        if action is None:
            # Fallback: dummy small random action with plausible dim
            self.logger.info("No mapper loaded or mapper failed, generating dummy action")
            if hasattr(packet, 'qpos') and packet.qpos is not None:
                action_dim = len(packet.qpos)
            else:
                action_dim = 7
            action = np.random.uniform(-0.1, 0.1, action_dim).tolist()
        
        
        # Log what we're doing
        self.logger.debug(f"Generating action for robot {robot_id}, mission: {mission}")
        self.logger.debug(f"Action: {action}")
        
        # Fill the action in the packet
        packet.action = action
        time_taken = time.time() - time_now
        self.logger.debug(f"Action generated for robot {robot_id} in {time_taken:.2f} seconds")
        
        return packet
            