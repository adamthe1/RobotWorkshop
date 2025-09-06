#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logger_config import get_logger
from control_panel.missions import SUPPORTED_MISSIONS

load_dotenv()


class EpisodeActionMapper:
    """
    Minimal episode action replayer that serves actions from a saved parquet.

    Assumptions (matching our recorder):
    - Parquet has an 'action' column with per-frame joint target arrays (list-like)
    - Action dimensionality matches the robot actuators used by the server
      (RobotBodyControl applies by filtering actuators with name prefix robot_id_)

    Basic use:
    - mapper = EpisodeActionMapper(path)
    - action = mapper.next_action(robot_id, joint_names, qpos, qvel)

    This class is intentionally simple and deterministic to be safe.
    """

    def __init__(self, robot_types: List[str]):
        self.logger = get_logger('EpisodeActionMapper')
        self.types_to_paths = {
            "FrankaPanda": {"pour beer": "finetuning/example_pouring/data/chunk-000/episode_000001.parquet"},
            "SO101": {"pour beer": "finetuning/so101_finetune/data/chunk-000/episode_000001.parquet"},
            # Add more robot types and their episode paths here
        }
        self.main_dir = os.getenv("MAIN_DIRECTORY", "/home/adam/Documents/coding/autonomous")
        self.slicing = 1
        self.loop = False
        self.speed = 6.0
        self.episode_actions = {}
        self.episode_lengths = {}

        for rt in robot_types:
            missions = self.types_to_paths.get(rt, {})
            for mission, ep_path in missions.items():
                if mission not in SUPPORTED_MISSIONS:
                    self.logger.warning(f"Mission '{mission}' for robot type '{rt}' is not supported.")
                ep_path = os.path.join(self.main_dir, ep_path)
                self.logger.info(f"Loading episode for robot type {rt} and mission {mission} from {ep_path}")
                if ep_path:
                    self.episode_actions[(rt, mission)] = self._load_episode(ep_path)
                self.episode_lengths[(rt, mission)] = len(self.episode_actions[(rt, mission)])

        self.robot_id_dict = {}

    
    def _load_episode(self, p: str) -> None:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Parquet file not found: {p}")
        self.df = pd.read_parquet(p)
        if 'action' not in self.df.columns:
            raise ValueError("Parquet missing 'action' column")
        # Normalize actions to np.ndarray of shape (T, A)
        actions = self.df['action'].tolist()
        if self.slicing > 1:
            actions = actions[::self.slicing]
       # Filter actions to ensure all are the same length
        filtered = []
        expected_dim = None
        for i, act in enumerate(actions):
            arr = np.array(act, dtype=float).ravel()
            if expected_dim is None:
                expected_dim = len(arr)
            if len(arr) == expected_dim:
                filtered.append(arr)
            else:
                self.logger.warning(f"Skipping action {i} (dim {len(arr)} != {expected_dim})")
        if not filtered:
            self.logger.error("No valid actions after filtering")
            return None
        actions = np.stack(filtered)
        if self.speed != 1.0:
            actions = self._time_resample(actions, self.speed)
        return actions
    
    def reset(self, robot_id: Optional[str] = None):
        """Reset cursor to beginning - robot_id parameter ignored for compatibility"""
        result = self.robot_id_dict.pop(robot_id, None)
        if result is None:
            self.logger.warning(f"Reset called for unknown robot_id: {robot_id}")

    def next_action(self, robot_id: str, robot_type: str, mission: str) -> List[float]:
        """
        Return the next action - robot_id parameter ignored for compatibility.
        
        For now this replays sequentially. It does not yet align by state/timestamp,
        but that can be added (e.g., nearest-neighbor on state or by stored timestamps).
        """
        self.logger.debug(f"Next action requested for robot_id: {robot_id}, type: {robot_type}, mission: {mission}")
        if robot_id not in self.robot_id_dict:
            self.robot_id_dict[robot_id] = 0
        idx = self.robot_id_dict[robot_id]
        self.robot_id_dict[robot_id] = idx + 1
        if idx >= self.episode_lengths.get((robot_type, mission), 0):
            if self.loop and self.episode_lengths.get((robot_type, mission), 0) > 0:
                idx = 0
            else:
                # No more actions; return zeros with appropriate length if possible
                length = len(self.actions[0]) if self.episode_lengths.get((robot_type, mission), 0) > 0 else 0
                return [0.0] * length

        act = self.episode_actions[(robot_type, mission)][idx]
        self.logger.debug(f"Action for robot_id {robot_id} at idx {idx}: {act}")
        # Convert to flat list of floats
        if isinstance(act, np.ndarray):
            act_out = act.astype(float).ravel().tolist()
        else:
            act_out = [float(x) for x in act]
            
        return act_out

    def get_progress(self, robot_id: str, robot_type: str, mission: str) -> float:
        """Return progress through the episode as a float in [0, 1]."""
        if robot_id not in self.robot_id_dict:
            self.logger.warning(f"Progress requested for unknown robot_id: {robot_id}")
            return 0.0
        return min(self.robot_id_dict[robot_id] / max(1, self.episode_lengths.get((robot_type, mission), 0)), 1.0)
    
    def load_type(self, robot_type: str) -> bool:
        """Load type for compatibility - no-op."""
        return True

    def _time_resample(
        self,
        frames: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """
        Resample along time to implement playback speed.
        - frames: ndarray (T, A)
        - speed > 1.0: slower (more frames, interpolated)
        - speed < 1.0: faster (fewer frames, skip)
        """
        if speed == 1.0 or frames.shape[0] == 0:
            return frames

        T, A = frames.shape
        new_T = max(1, int(round(T / speed)))
        if new_T == T:
            return frames

        old_t = np.linspace(0.0, 1.0, num=T, endpoint=True)
        new_t = np.linspace(0.0, 1.0, num=new_T, endpoint=True)
        out = np.empty((new_T, A), dtype=float)
        for d in range(A):
            out[:, d] = np.interp(new_t, old_t, frames[:, d])
        return out