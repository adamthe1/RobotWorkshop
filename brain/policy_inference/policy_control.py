from .episode_action_mapper import EpisodeActionMapper
from .joint_test_mapper import JointTestMapper
from .policy_embodiment_manager import PolicyEmbodimentManager
from .preprocessing import Preprocessing
from logger_config import get_logger
from typing import List, Dict, Any
import numpy as np
import time


class PolicyManager:
    def __init__(self, robot_dict: Dict[str, str] = None):
        self.logger = get_logger('PolicyManager')
        self.robot_dict = robot_dict or {}
        robot_types = set(self.robot_dict.values())
        self.logger.info(f"Initializing PolicyManager with robot types: {robot_types}")

        # Map robot types to policy names
        self.type_to_policy = {"FrankaPanda": "Episode", "SO101": "Episode"}  # Extend as needed

        # Find which policies are needed
        needed_policy_names = set(self.type_to_policy[typ] for typ in robot_types if typ in self.type_to_policy)
        self.logger.info(f"Policies needed: {needed_policy_names}")

        # Initialize only needed policies, passing all robot types
        self.policies = {}
        for name in needed_policy_names:
            if name == "Episode":
                # Only MAIN_DIRECTORY is used; episodes resolved by type
                self.policies[name] = EpisodeActionMapper(list(robot_types))
            elif name == "JointTest":
                self.policies[name] = JointTestMapper(list(robot_types))
            # Add more policies here as needed

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



class PolicyInference:
    def __init__(self, robot_dict):
        self.robot_dict = robot_dict
        self.logger = get_logger('PolicyInference')
        self.embodiment_manager = PolicyEmbodimentManager(robot_dict=robot_dict)
        self.policy_manager = PolicyManager(robot_dict=robot_dict)
        self.preprocesser = Preprocessing(self.policy_manager.policies)

    def fill_action(self, packet):
        time_now = time.time()
        robot_id = packet.robot_id
        mission = packet.mission
        submission = packet.submission

        try:
            policy_item = self.preprocesser.preprocess(packet)
            action_reply = self.get_policy_action(policy_item, robot_id)
            action = action_reply.get('action', None)
            mission_status = action_reply.get('mission_status', None)

        except Exception as e:
            self.logger.error(f"Policy inference failed, falling back to dummy: {e}")
            action = None
            mission_status = None

        if action is None:
            # Fallback: dummy small random action with plausible dim
            self.logger.info("No mapper loaded or mapper failed, generating dummy action")
            action = self.get_dummy_action(robot_id)

        # Fill the action in the packet
        packet.action = action
        if mission_status is not None:
            packet.mission_status = mission_status
        
        time_taken = time.time() - time_now
        self.logger.debug(f"Action generated for robot {robot_id} in {time_taken:.2f} seconds")
        
        return packet

    def get_policy_action(self, policy_item, robot_id):
        # Implement action inference logic based on the observation
        policy_name, policy = self.policy_manager.get_policy_for_robot_id(robot_id)
        self.logger.debug(f"Using policy '{policy_name}' for robot {robot_id} of type {self.robot_dict.get(robot_id, 'Unknown')}")
        
        reply = {'action': None, 'mission_status': None}
        if policy_name == 'Episode':
            robot_type = self.robot_dict.get(robot_id, None)
            mission_name = policy_item.get('prompt', '')
            reply['action'] = policy.next_action(robot_id, robot_type, mission_name)
            progress = policy.get_progress(robot_id, robot_type, mission_name)
            # If mapper is serving a reset tail, don't mark completed yet
            reset_pending = False
            if hasattr(policy, 'is_reset_pending'):
                try:
                    reset_pending = policy.is_reset_pending(robot_id, robot_type, mission_name)
                except Exception:
                    reset_pending = False
            self.logger.debug(f"REPLAY: Robot {robot_id} progress: {progress*100:.1f}% reset_pending={reset_pending} action: {reply['action']}")
            if progress == 1.0 and not reset_pending:
                reply['mission_status'] = 'completed'
                self.logger.info(f"REPLAY: Robot {robot_id} mission completed")
                policy.reset(robot_id)
        return reply

    def get_dummy_action(self, robot_id):
        action_dim = len(self.embodiment_manager.get_robot_actuator_list(robot_id))
        return np.random.uniform(-0.1, 0.1, action_dim).tolist()       
