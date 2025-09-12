#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logger_config import get_logger
from control_panel.missions import SUPPORTED_MISSIONS

load_dotenv()


class EpisodeActionMapper:
    """
    Clean episode action replayer with observation state loading and action continuity.
    Supports interpolation for speedup and maintains state-based action progression.
    """

    def __init__(self, robot_types: List[str]):
        self.logger = get_logger('EpisodeActionMapper')
        self.main_dir = os.getenv("MAIN_DIRECTORY", os.getcwd())
        base_path = os.path.join(self.main_dir, "finetuning", "mission_episodes")
        self.types_to_paths = self._discover_mission_paths(robot_types, base_path)

        # Configuration
        self.loop = False
        self.replay_speed = self._get_float_env("REPLAY_SPEED", 1.0)
        self.use_obs_as_action = os.getenv("REPLAY_USE_OBS_AS_ACTION", "0").strip().lower() in ("1", "true", "yes", "on")
        self.state_tolerance = self._get_float_env("STATE_TOLERANCE", 0.1)
        self.max_continuity_frames = int(os.getenv("MAX_CONTINUITY_FRAMES", "10"))

        # Storage
        self.episode_actions: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.episode_lengths: Dict[Tuple[str, str, str], int] = {}
        self.mission_sub_missions: Dict[Tuple[str, str], List[str]] = {}
        self.observation_states: Dict[Tuple[str, str, str], np.ndarray] = {}

        # Load episodes for each robot type and mission
        self._load_all_episodes()

        # Robot state tracking
        self.robot_mission_state: Dict[Tuple[str, str, str], dict] = {}
        self.robot_current_qpos: Dict[str, np.ndarray] = {}
        self.continuity_counters: Dict[Tuple[str, str, str], int] = {}
        self._reset_tails: Dict[Tuple[str, str, str], dict] = {}

    def _get_float_env(self, env_var: str, default: float) -> float:
        """Safely get float from environment variable."""
        try:
            return float(os.getenv(env_var, str(default)))
        except (ValueError, TypeError):
            return default

    def _load_all_episodes(self) -> None:
        """Load all episode data for configured robot types and missions."""
        for robot_type, missions in self.types_to_paths.items():
            for mission, mission_folder in missions.items():
                self._load_mission_episodes(robot_type, mission, mission_folder)

    def _load_mission_episodes(self, robot_type: str, mission: str, mission_folder: str) -> None:
        """Load all sub-mission episodes for a specific robot type and mission."""
        sub_missions = SUPPORTED_MISSIONS.get(mission, [])
        if not sub_missions:
            self.logger.warning(f"No sub-missions configured for mission '{mission}'")
            return

        self.mission_sub_missions[(robot_type, mission)] = sub_missions
        
        for sub_mission in sub_missions:
            self._load_single_episode(robot_type, mission, sub_mission, mission_folder)

    def _load_single_episode(self, robot_type: str, mission: str, sub_mission: str, mission_folder: str) -> None:
        """Load a single episode file and process it."""
        parquet_path = os.path.join(mission_folder, f"{sub_mission}.parquet")
        
        if not os.path.isfile(parquet_path):
            self._handle_missing_episode(robot_type, mission, sub_mission)
            return

        try:
            # Load actions
            if self.use_obs_as_action:
                actions = self._build_actions_from_observation(mission_folder, parquet_path)
            else:
                actions = self._load_parquet_actions(parquet_path)

            if actions is None or actions.size == 0:
                self.logger.warning(f"No valid actions found for {robot_type}/{mission}/{sub_mission}")
                return

            # Apply speed resampling if needed
            if self.replay_speed != 1.0 and actions.shape[0] > 1:
                actions = self._resample_actions_with_interpolation(actions, self.replay_speed)

            # Store processed actions
            episode_key = (robot_type, mission, sub_mission)
            self.episode_actions[episode_key] = actions
            self.episode_lengths[episode_key] = int(actions.shape[0])
            
            self.logger.info(f"Loaded {robot_type}/{mission}/{sub_mission}: frames={actions.shape[0]} dim={actions.shape[1]}")

        except Exception as e:
            self.logger.error(f"Failed to load episode {robot_type}/{mission}/{sub_mission}: {e}")

    def _handle_missing_episode(self, robot_type: str, mission: str, sub_mission: str) -> None:
        """Handle missing episode files, potentially creating synthetic reset sequences."""
        if sub_mission.replace(" ", "_") in ("reset_before_new_mission", "reset", "return_to_initial"):
            reset_actions = self._build_synthetic_reset(robot_type, mission)
            if reset_actions is not None:
                episode_key = (robot_type, mission, sub_mission)
                self.episode_actions[episode_key] = reset_actions
                self.episode_lengths[episode_key] = int(reset_actions.shape[0])
                self.logger.info(f"Created synthetic reset for {robot_type}/{mission}/{sub_mission}: frames={reset_actions.shape[0]}")
        else:
            self.logger.warning(f"Missing parquet for {robot_type}/{mission}/{sub_mission}")

    def _build_actions_from_observation(self, mission_folder: str, parquet_path: str) -> Optional[np.ndarray]:
        """Build action sequences from observation state data."""
        try:
            df = pd.read_parquet(parquet_path)
            if 'observation.state' not in df.columns:
                self.logger.warning(f"No observation.state column in {parquet_path}")
                return None

            # Load mapping configuration
            mapping_config = self._load_observation_mapping(mission_folder)
            if mapping_config is None:
                return None

            # Extract and map observation states
            observations = df['observation.state'].tolist()
            actions = []
            
            for obs in observations:
                obs_array = np.asarray(obs, dtype=float).ravel()
                if obs_array.size > max(mapping_config['indices']):
                    action_values = obs_array[mapping_config['indices']]
                    actions.append(action_values)

            if not actions:
                self.logger.warning(f"No valid observations mapped in {parquet_path}")
                return None

            actions_array = np.stack(actions, axis=0)
            
            # Add gripper control if needed
            if mapping_config['include_gripper'] and actions_array.shape[1] < mapping_config['expected_dim']:
                actions_array = self._add_gripper_control(actions_array, mapping_config['expected_dim'])

            return actions_array

        except Exception as e:
            self.logger.error(f"Failed to build actions from observation: {e}")
            return None

    def _load_observation_mapping(self, mission_folder: str) -> Optional[dict]:
        """Load observation to action mapping configuration."""
        config_path = os.path.join(mission_folder, "obs_to_action.json")
        
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                indices = config.get('indices', list(range(7)))
                return {
                    'indices': indices,
                    'include_gripper': config.get('include_gripper', True),
                    'expected_dim': config.get('expected_dim', 8)
                }
            except Exception as e:
                self.logger.warning(f"Failed to load obs_to_action.json: {e}")
        
        # Default mapping for Franka Panda
        return {
            'indices': list(range(7)),
            'include_gripper': True,
            'expected_dim': 8
        }

    def _add_gripper_control(self, actions: np.ndarray, target_dim: int) -> np.ndarray:
        """Add gripper control column to actions array."""
        gripper_values = np.zeros((actions.shape[0], target_dim - actions.shape[1]))
        return np.hstack([actions, gripper_values])

    def _discover_mission_paths(self, robot_types: List[str], base_path: str) -> Dict[str, Dict[str, str]]:
        """Discover available mission paths for each robot type."""
        mission_paths = {}
        
        for robot_type in robot_types:
            robot_type_dir = os.path.join(base_path, robot_type)
            if not os.path.isdir(robot_type_dir):
                self.logger.warning(f"Robot type directory not found: {robot_type_dir}")
                continue
            
            available_missions = {}
            for mission in SUPPORTED_MISSIONS:
                mission_dir = os.path.join(robot_type_dir, mission)
                if os.path.isdir(mission_dir):
                    available_missions[mission] = mission_dir
            
            if available_missions:
                mission_paths[robot_type] = available_missions
        
        return mission_paths

    # -------- Parquet loading --------
    def _load_parquet_actions(self, path: str) -> Optional[np.ndarray]:
        """Load 'action' column and return ndarray(T, A)."""
        df = pd.read_parquet(path)
        if 'action' not in df.columns:
            self.logger.error(f"Missing 'action' column in {path}")
            return None
        
        actions = df['action'].tolist()
        action_arrays = [np.asarray(a, dtype=float) for a in actions]
        return np.stack(action_arrays, axis=0)

    def _resample_actions_with_interpolation(self, actions: np.ndarray, speed: float) -> np.ndarray:
        """Resample actions with interpolation for speed changes."""
        if speed == 1.0 or actions.shape[0] <= 1:
            return actions
        
        original_frames = actions.shape[0]
        target_frames = max(1, int(original_frames / speed))
        
        if speed > 1.0:  # Speed up
            indices = np.linspace(0, original_frames - 1, target_frames)
            return np.array([actions[int(round(idx))] for idx in indices])
        else:  # Slow down
            indices = np.linspace(0, original_frames - 1, target_frames)
            return np.array([self._interpolate_action(actions, idx) for idx in indices])
    
    def _interpolate_action(self, actions: np.ndarray, position: float) -> np.ndarray:
        """Linear interpolation between action frames."""
        idx_low = int(np.floor(position))
        idx_high = min(idx_low + 1, actions.shape[0] - 1)
        alpha = position - idx_low
        
        return (1 - alpha) * actions[idx_low] + alpha * actions[idx_high]

    def _build_synthetic_reset(self, robot_type: str, mission: str) -> Optional[np.ndarray]:
        """Create a simple reset sequence to initial joint positions."""
        saved_state = self._load_latest_robot_state()
        if saved_state is None:
            self.logger.warning("No saved robot state found for reset")
            return None
        
        joint_names, qpos = saved_state
        target_joint_positions = self._extract_franka_joints(joint_names, qpos)
        
        if target_joint_positions is None:
            self.logger.warning("Could not extract Franka joint positions")
            return None
        
        # Add gripper (open by default)
        reset_action = np.append(target_joint_positions, 0.0)
        
        # Create hold frames
        hold_duration = self._get_float_env("RESET_HOLD_SECONDS", 2.0)
        control_frequency = self._get_float_env("CONTROL_HZ", 60.0)
        num_frames = max(1, int(round(hold_duration * control_frequency)))
        
        return np.tile(reset_action, (num_frames, 1))
    
    def _load_latest_robot_state(self) -> Optional[tuple]:
        """Load the most recent robot state from saved states."""
        import glob
        
        search_dirs = [
            os.getenv("REPLAY_SAVED_STATE_DIR", ""),
            os.path.join(self.main_dir, "finetuning", "saved_robot_states")
        ]
        
        state_files = []
        for search_dir in search_dirs:
            if search_dir and os.path.isdir(search_dir):
                state_files.extend(glob.glob(os.path.join(search_dir, "*.npz")))
        
        if not state_files:
            return None
        
        latest_file = max(state_files, key=os.path.getmtime)
        try:
            state_data = np.load(latest_file, allow_pickle=True)
            return state_data["joint_names"], state_data["qpos"]
        except Exception as e:
            self.logger.error(f"Failed to load robot state from {latest_file}: {e}")
            return None
    
    def _extract_franka_joints(self, joint_names: list, qpos: np.ndarray) -> Optional[np.ndarray]:
        """Extract Franka Panda joint positions from general joint data."""
        franka_joints = []
        
        for i in range(1, 8):
            joint_name = f"joint{i}"
            try:
                joint_idx = list(joint_names).index(joint_name)
                franka_joints.append(float(qpos[joint_idx]))
            except ValueError:
                franka_joints.append(0.0)
        
        return np.array(franka_joints)

      # -------- Control flow --------
    def reset(self, robot_id: Optional[str] = None, robot_type: Optional[str] = None, mission: Optional[str] = None):
        """Reset robot mission state to beginning."""
        if all([robot_id, robot_type, mission]):
            state_key = (robot_id, robot_type, mission)
            self.robot_mission_state[state_key] = {'i_sub': 0, 'i_frame': 0}
            self.continuity_counters[state_key] = 0
        else:
            self.robot_mission_state.clear()
            self.continuity_counters.clear()

    def get_first_episode_frame(self, robot_type: str, mission: str) -> Optional[np.ndarray]:
        """Get the first frame of the first sub-mission episode."""
        sub_missions = self.mission_sub_missions.get((robot_type, mission), [])
        if not sub_missions:
            return None
        
        first_sub_mission = sub_missions[0]
        episode_key = (robot_type, mission, first_sub_mission)
        first_actions = self.episode_actions.get(episode_key)
        
        if first_actions is not None and first_actions.shape[0] > 0:
            return first_actions[0]
        
        return None

    def update_robot_qpos(self, robot_id: str, qpos: np.ndarray) -> None:
        """Update current robot joint positions for continuity checking."""
        self.robot_current_qpos[robot_id] = np.asarray(qpos, dtype=float)

    def is_state_close_to_action(self, robot_id: str, action: np.ndarray) -> bool:
        """Check if current robot state is close to target action."""
        if robot_id not in self.robot_current_qpos:
            return False
        
        current_qpos = self.robot_current_qpos[robot_id]
        target_qpos = action[:7]  # First 7 dimensions are joint positions
        
        distance = np.linalg.norm(current_qpos - target_qpos)
        return distance <= self.state_tolerance

    def next_action(self, robot_id: str, robot_type: str, mission: str) -> List[float]:
        """Get next action with continuity checking based on current robot state."""
        
        # Handle auto-reset mission
        if mission == "__auto_reset__":
            return self._handle_auto_reset(robot_id, robot_type, mission)
        
        # Initialize state if needed
        state_key = (robot_id, robot_type, mission)
        if state_key not in self.robot_mission_state:
            self.robot_mission_state[state_key] = {'i_sub': 0, 'i_frame': 0}
            self.continuity_counters[state_key] = 0
        
        current_state = self.robot_mission_state[state_key]
        
        # Get current action from episode
        current_action = self._get_current_episode_action(robot_id, robot_type, mission)
        if current_action is None:
            return self.get_zero_action(robot_type, mission)
        
        # Check continuity - if robot state is close enough, advance to next action
        if self._should_advance_action(state_key, current_action):
            self._advance_to_next_frame(robot_id, robot_type, mission)
            return self.next_action(robot_id, robot_type, mission)
        
        # If not close enough, continue with current action
        return current_action.tolist()
    
    def _handle_auto_reset(self, robot_id: str, robot_type: str, mission: str) -> List[float]:
        """Handle auto-reset mission with synthetic reset sequence."""
        reset_key = (robot_id, robot_type, mission)
        reset_sequence = self._reset_tails.get(reset_key)
        
        if reset_sequence is None:
            reset_actions = self._build_synthetic_reset(robot_type, mission)
            if reset_actions is not None:
                self._reset_tails[reset_key] = {'frames': reset_actions, 'idx': 0}
                reset_sequence = self._reset_tails[reset_key]
        
        if reset_sequence is not None:
            frames = reset_sequence['frames']
            idx = reset_sequence['idx']
            if idx < frames.shape[0]:
                action = frames[idx].astype(float).ravel()
                reset_sequence['idx'] = idx + 1
                return action.tolist()
            self._reset_tails.pop(reset_key, None)
        
        return self.get_zero_action(robot_type, mission)
    
    def _get_current_episode_action(self, robot_id: str, robot_type: str, mission: str) -> Optional[np.ndarray]:
        """Get the current action from the episode sequence."""
        state_key = (robot_id, robot_type, mission)
        current_state = self.robot_mission_state[state_key]
        
        sub_missions = self.mission_sub_missions.get((robot_type, mission), [])
        if not sub_missions:
            return None
        
        # Advance through sub-missions if needed
        while current_state['i_sub'] < len(sub_missions):
            sub_mission = sub_missions[current_state['i_sub']]
            episode_key = (robot_type, mission, sub_mission)
            actions = self.episode_actions.get(episode_key)
            
            if actions is None or actions.shape[0] == 0:
                current_state['i_sub'] += 1
                current_state['i_frame'] = 0
                continue
            
            if current_state['i_frame'] < actions.shape[0]:
                return actions[current_state['i_frame']]
            else:
                current_state['i_sub'] += 1
                current_state['i_frame'] = 0
        
        # Try reset sequence if episodes are complete
        return self._get_reset_sequence_action(robot_id, robot_type, mission)
    
    def _get_reset_sequence_action(self, robot_id: str, robot_type: str, mission: str) -> Optional[np.ndarray]:
        """Get action from reset sequence if available."""
        reset_key = (robot_id, robot_type, mission)
        reset_sequence = self._reset_tails.get(reset_key)
        
        if reset_sequence is None:
            reset_actions = self._build_synthetic_reset(robot_type, mission)
            if reset_actions is not None:
                self._reset_tails[reset_key] = {'frames': reset_actions, 'idx': 0}
                reset_sequence = self._reset_tails[reset_key]
        
        if reset_sequence is not None:
            frames = reset_sequence['frames']
            idx = reset_sequence['idx']
            if idx < frames.shape[0]:
                return frames[idx]
            self._reset_tails.pop(reset_key, None)
        
        return None
    
    def _should_advance_action(self, state_key: tuple, current_action: np.ndarray) -> bool:
        """Determine if we should advance to the next action based on robot state."""
        robot_id = state_key[0]
        
        # Check if we've exceeded continuity limit
        if self.continuity_counters.get(state_key, 0) >= self.max_continuity_frames:
            self.continuity_counters[state_key] = 0
            return True
        
        # Check if robot state is close to target action
        if self.is_state_close_to_action(robot_id, current_action):
            self.continuity_counters[state_key] = 0
            return True
        
        # Increment continuity counter
        self.continuity_counters[state_key] = self.continuity_counters.get(state_key, 0) + 1
        return False
    
    def _advance_to_next_frame(self, robot_id: str, robot_type: str, mission: str) -> None:
        """Advance to the next frame in the episode sequence."""
        state_key = (robot_id, robot_type, mission)
        current_state = self.robot_mission_state[state_key]
        
        current_state['i_frame'] += 1
        
        # If we've completed all frames, handle completion
        sub_missions = self.mission_sub_missions.get((robot_type, mission), [])
        if (current_state['i_sub'] >= len(sub_missions) - 1 and 
            current_state['i_frame'] >= self._get_current_episode_length(robot_type, mission, sub_missions[-1])):
            
            if self.loop:
                current_state['i_sub'] = 0
                current_state['i_frame'] = 0
    
    def _get_current_episode_length(self, robot_type: str, mission: str, sub_mission: str) -> int:
        """Get the length of the current episode."""
        episode_key = (robot_type, mission, sub_mission)
        return self.episode_lengths.get(episode_key, 0)

    def get_progress(self, robot_id: str, robot_type: str, mission: str) -> float:
        """Calculate mission progress as a fraction between 0.0 and 1.0."""
        if mission == "__auto_reset__":
            return self._get_auto_reset_progress(robot_id, robot_type, mission)
        
        return self._get_episode_progress(robot_id, robot_type, mission)
    
    def _get_auto_reset_progress(self, robot_id: str, robot_type: str, mission: str) -> float:
        """Calculate progress for auto-reset mission."""
        reset_sequence = self._reset_tails.get((robot_id, robot_type, mission))
        if not reset_sequence:
            return 1.0
        
        frames = reset_sequence.get('frames')
        current_frame = int(reset_sequence.get('idx', 0))
        
        if not isinstance(frames, np.ndarray) or frames.shape[0] == 0:
            return 1.0
        
        return min(current_frame / frames.shape[0], 1.0)
    
    def _get_episode_progress(self, robot_id: str, robot_type: str, mission: str) -> float:
        """Calculate progress for regular episode-based mission."""
        state_key = (robot_id, robot_type, mission)
        current_state = self.robot_mission_state.get(state_key)
        sub_missions = self.mission_sub_missions.get((robot_type, mission), [])
        
        if not current_state or not sub_missions:
            return 0.0
        
        total_frames = 0
        completed_frames = 0
        
        for i, sub_mission in enumerate(sub_missions):
            episode_key = (robot_type, mission, sub_mission)
            episode_length = self.episode_lengths.get(episode_key, 0)
            total_frames += episode_length
            
            if i < current_state['i_sub']:
                completed_frames += episode_length
            elif i == current_state['i_sub']:
                completed_frames += min(current_state['i_frame'], episode_length)
        
        if total_frames == 0:
            return 0.0
        
        progress = min(completed_frames / total_frames, 1.0)
        
        # Keep progress below 1.0 if reset sequence is pending
        if self._has_pending_reset_sequence(robot_id, robot_type, mission):
            progress = min(progress, 0.999)
        
        return progress
    
    def _has_pending_reset_sequence(self, robot_id: str, robot_type: str, mission: str) -> bool:
        """Check if there's a pending reset sequence for this robot."""
        reset_sequence = self._reset_tails.get((robot_id, robot_type, mission))
        if not reset_sequence:
            return False
        
        frames = reset_sequence.get('frames')
        current_frame = int(reset_sequence.get('idx', 0))
        
        return isinstance(frames, np.ndarray) and current_frame < frames.shape[0]

    def load_type(self, robot_type: str) -> bool:
        """Legacy method for compatibility."""
        return True

    def get_zero_action(self, robot_type: str, mission: str) -> List[float]:
        """Get a zero action vector with appropriate dimensions."""
        # Try to infer dimensions from loaded episodes
        for episode_key, actions in self.episode_actions.items():
            if episode_key[0] == robot_type and actions is not None and actions.size > 0:
                return [0.0] * int(actions.shape[1])
        
        # Fallback to default Franka Panda dimensions
        return [0.0] * 8
