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
        self.main_dir = os.getenv("MAIN_DIRECTORY", "/home/adam/Documents/coding/autonomous")
        mission_episode_paths = os.path.join(self.main_dir, "finetuning/mission_episodes")
        self.types_to_missions_to_submissions = self.load_submission_paths(robot_types, mission_episode_paths)
        self.build_actions_from_obs = True
        self.slicing = 1
        self.loop = False
        self.speed = 1.0
        self.episode_actions = {}
        self.episode_lengths = {}

        for rt in robot_types:
            missions = self.types_to_missions_to_submissions.get(rt, {})
            for mission, submissions in missions.items():
                if mission not in SUPPORTED_MISSIONS:
                    self.logger.warning(f"Mission '{mission}' for robot type '{rt}' is not supported.")
                for sub_mission, ep_path in submissions.items():
                    self.logger.info(f"Loading episode for robot type {rt}, mission {mission}, sub-mission {sub_mission} from {ep_path}")
                    if ep_path:
                        try:
                            self.episode_actions[(rt, mission, sub_mission)] = self._load_episode(ep_path)
                            self.episode_lengths[(rt, mission, sub_mission)] = len(self.episode_actions[(rt, mission, sub_mission)])
                        except Exception as e:
                            self.logger.error(f"Failed to load episode for robot type {rt}, mission {mission}, sub-mission {sub_mission}: {e}")
                            raise

        self.robot_id_dict = {}

    def load_submission_paths(self, robot_types: List[str], base_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load mapping of robot types to missions to episode parquet paths."""
        type_to_missions_to_submissions = {}
        for rt in robot_types:
            rt_path = os.path.join(base_path, rt)
            if not os.path.isdir(rt_path):
                self.logger.warning(f"Robot type directory not found: {rt_path}")
                continue
            missions = {}
            for mission, sub_missions in SUPPORTED_MISSIONS.items():
                submissions = {}
                mission_path = os.path.join(rt_path, mission)
                if not os.path.isdir(mission_path):
                    self.logger.warning(f"Mission directory not found for {rt}: {mission_path}")
                    continue
                for sub_mission in sub_missions:
                    sub_mission_file = f"{sub_mission}.parquet"
                    sub_mission_path = os.path.join(mission_path, sub_mission_file)
                    if not os.path.isfile(sub_mission_path):
                        self.logger.warning(f"Mission file not found for {rt}: {sub_mission_path}")
                        continue
                    submissions[sub_mission] = sub_mission_path
                if submissions:
                    missions[mission] = submissions
            if missions:
                type_to_missions_to_submissions[rt] = missions
        return type_to_missions_to_submissions
    
    def _load_episode(self, p: str) -> np.ndarray:
        if not os.path.isfile(p):
            error_msg = f"Parquet file not found: {p}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.build_actions_from_obs:
            actions = self._try_build_actions_from_obs(
                parquet_path=p,
                expected_dim=8  # Assuming 7 joints + 1 gripper
            )
        else:
            self.logger.warning(f"Falling back to 'action' column for episode at {p}")
            df = pd.read_parquet(p)
            if 'action' not in df.columns:
                error_msg = "Parquet missing 'action' column"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            # Normalize actions to np.ndarray of shape (T, A)
            actions = df['action'].tolist()
        
        if actions is None:
            error_msg = f"Failed to load actions for episode at {p}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
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
            error_msg = "No valid actions after filtering"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        actions = np.stack(filtered)
        if self.speed != 1.0:
            actions = self._time_resample(actions, self.speed)
        return actions

    def _try_build_actions_from_obs(
        self,
        parquet_path: str,
        expected_dim: int
    ) -> Optional[np.ndarray]:
        """
        Construct an action matrix from the 'observation.state' column in a Parquet file.

        Behavior:
        - Uses the first N elements of each observation vector as the robot joint values,
        where N = int(os.getenv("REPLAY_OBS_FIRST_N", "7")).
        - Skips rows whose observation vector is too short.
        - If 'expected_dim' is larger than the chosen dimension:
            * If 'action' column exists, append the (expected_dim-1)-th value from it
            (commonly the gripper channel). Fallback = 0.0 if unavailable.
            * Otherwise, pad the missing dimensions with zeros.

        Returns:
            np.ndarray of shape (T, expected_dim) on success, or None on failure.
        """
        try:
            # ---------- Load the parquet ----------
            df = pd.read_parquet(parquet_path)
            if "observation.state" not in df.columns:
                self.logger.warning(f"{parquet_path} has no 'observation.state' column")
                return None

            # ---------- Select indices (always heuristic now) ----------
            n_joints = self.replay_first_n_joints = 7
            indices = list(range(n_joints))

            # ---------- Extract per-row actions ----------
            obs_series = df["observation.state"].tolist()
            action_rows: List[np.ndarray] = []

            for obs_row in obs_series:
                obs_vec = np.asarray(obs_row, dtype=float).ravel()
                if obs_vec.size <= max(indices):
                    continue
                action_rows.append(obs_vec[indices])

            if not action_rows:
                self.logger.warning(
                    f"{parquet_path}: could not map any observation rows with indices {indices}"
                )
                return None

            actions = np.stack(action_rows, axis=0)

            # ---------- Ensure output has expected_dim ----------
            current_dim = actions.shape[1]
            if expected_dim > current_dim:
                missing = expected_dim - current_dim

                if "action" in df.columns:
                    # Try to extract gripper (or similar) from original 'action' column
                    source_actions = df["action"].tolist()
                    appended_vals: List[float] = []
                    row_count = actions.shape[0]

                    for raw in source_actions[:row_count]:
                        a = np.asarray(raw, dtype=float).ravel()
                        if a.size >= expected_dim:
                            appended_vals.append(float(a[expected_dim - 1]))
                        else:
                            appended_vals.append(0.0)

                    actions = np.hstack(
                        [actions, np.array(appended_vals, dtype=float).reshape(-1, 1)]
                    )

                    # If still short, pad with zeros
                    if actions.shape[1] < expected_dim:
                        pad_cols = expected_dim - actions.shape[1]
                        actions = np.hstack(
                            [actions, np.zeros((actions.shape[0], pad_cols), dtype=float)]
                        )
                else:
                    # No 'action' column: pad all missing dims with zeros
                    actions = np.hstack(
                        [actions, np.zeros((actions.shape[0], missing), dtype=float)]
                    )

            self.logger.info(
                f"Using observation.state as actions via first_{len(indices)}_obs; dim={actions.shape[1]}"
            )
            return actions

        except Exception as e:
            self.logger.warning(f"Failed building actions from obs for {parquet_path}: {e}")
            return None

    def reset(self, robot_id: Optional[str] = None):
        """Reset cursor to beginning - robot_id parameter ignored for compatibility"""
        result = self.robot_id_dict.pop(robot_id, None)
        if result is None:
            self.logger.warning(f"Reset called for unknown robot_id: {robot_id}")

    def next_action(self, robot_id: str, robot_type: str, mission: str, submission: str) -> List[float]:
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
        
        episode_key = (robot_type, mission, submission)
        episode_length = self.episode_lengths.get(episode_key, 0)
        
        if idx >= episode_length:
            if self.loop and episode_length > 0:
                idx = 0
                self.robot_id_dict[robot_id] = 1
            else:
                # No more actions; return zeros with appropriate length if possible
                if episode_key in self.episode_actions and len(self.episode_actions[episode_key]) > 0:
                    length = len(self.episode_actions[episode_key][0])
                else:
                    length = 8  # fallback to expected_dim
                return [0.0] * length

        try:
            act = self.episode_actions[episode_key][idx]
        except KeyError:
            error_msg = f"Episode not found for key {episode_key}"
            self.logger.error(error_msg)
            raise KeyError(error_msg)
        except IndexError:
            error_msg = f"Action index {idx} out of range for episode {episode_key}"
            self.logger.error(error_msg)
            raise IndexError(error_msg)
        
        self.logger.debug(f"Action for robot_id {robot_id} at idx {idx}: {act}")
        # Convert to flat list of floats
        if isinstance(act, np.ndarray):
            act_out = act.astype(float).ravel().tolist()
        else:
            act_out = [float(x) for x in act]
            
        return act_out

    def get_progress(self, robot_id: str, robot_type: str, mission: str, submission: str) -> float:
        """Return progress through the episode as a float in [0, 1]."""
        if robot_id not in self.robot_id_dict:
            self.logger.warning(f"Progress requested for unknown robot_id: {robot_id}")
            return 0.0
        
        episode_key = (robot_type, mission, submission)
        episode_length = self.episode_lengths.get(episode_key, 1)
        return min(self.robot_id_dict[robot_id] / max(1, episode_length), 1.0)

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