#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logger_config import get_logger
from control_panel.missions import SUPPORTED_MISSIONS

load_dotenv()


class EpisodeActionMapper:
    """
    Deterministic episode action replayer for the new folder layout:
      finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet

    Exact-parity goal: returns the recorded action vectors as-is, with no
    remapping, scaling, or resampling. This ensures playback matches recording.
    """

    def __init__(self, robot_types: List[str]):
        self.logger = get_logger('EpisodeActionMapper')
        self.main_dir = os.getenv("MAIN_DIRECTORY", os.getcwd())
        base_path = os.path.join(self.main_dir, "finetuning", "mission_episodes")
        self.types_to_paths = self._discover_mission_paths(robot_types, base_path)

        # Controls
        self.loop: bool = False  # keep simple and deterministic
        # Time-warp without interpolation: >1.0 faster (skip), <1.0 slower (repeat)
        try:
            self.replay_speed: float = float(os.getenv("REPLAY_SPEED", "1.0"))
        except Exception:
            self.replay_speed = 1.0
        # Optional alternate source: derive actions from observation.state for parity
        self.use_obs_as_action: bool = os.getenv("REPLAY_USE_OBS_AS_ACTION", "0").strip() in ("1","true","yes","on")

        # Storage
        # {(rt, mission, sub_mission): np.ndarray(T,A)}
        self.episode_actions: Dict[tuple, np.ndarray] = {}
        self.episode_lengths: Dict[tuple, int] = {}
        # (rt, mission) -> ordered sub-missions list
        self.mission_sub_missions: Dict[tuple, List[str]] = {}

        # Load episodes for each robot type and mission
        for rt, missions in self.types_to_paths.items():
            for mission, mission_folder in missions.items():
                sub_missions = SUPPORTED_MISSIONS.get(mission, [])
                if not sub_missions:
                    self.logger.warning(f"No sub-missions configured for mission '{mission}'")
                    continue
                self.mission_sub_missions[(rt, mission)] = sub_missions
                for sm in sub_missions:
                    p = os.path.join(mission_folder, f"{sm}.parquet")
                    if not os.path.isfile(p):
                        # If this looks like a reset step, synthesize a reset sequence to initial state
                        if sm.replace(" ", "_") in ("reset_before_new_mission", "reset", "return_to_initial"):
                            arr = self._build_synthetic_reset(rt, mission)
                            if arr is not None:
                                key = (rt, mission, sm)
                                self.episode_actions[key] = arr
                                self.episode_lengths[key] = int(arr.shape[0])
                                self.logger.info(
                                    f"Synthesized reset sequence for {rt}/{mission}/{sm}: frames={arr.shape[0]} dim={arr.shape[1]}"
                                )
                                continue
                        self.logger.warning(f"Missing parquet for {rt}/{mission}/{sm}: {p}")
                        continue
                    arr = self._load_parquet_actions_exact(p)
                    # Optional: replace actions using observation.state with mapping
                    if self.use_obs_as_action:
                        mapped = self._try_build_actions_from_obs(mission_folder, p, expected_dim=int(arr.shape[1]))
                        if mapped is not None:
                            arr = mapped
                    # Optional time-warp (stepwise, no interpolation)
                    if self.replay_speed != 1.0 and arr.shape[0] > 1:
                        arr = self._time_resample_stepwise(arr, self.replay_speed)
                    key = (rt, mission, sm)
                    self.episode_actions[key] = arr
                    self.episode_lengths[key] = int(arr.shape[0])
                    self.logger.info(f"Loaded {rt}/{mission}/{sm}: frames={arr.shape[0]} dim={arr.shape[1]}")

        # Per-robot per-mission cursor
        # {(robot_id, robot_type, mission): {'i_sub': int, 'i_frame': int}}
        self.robot_mission_state: Dict[tuple, dict] = {}
        # Optional per-robot reset tails served after mission frames finish
        # {(robot_id, robot_type, mission): { 'frames': np.ndarray(T,A), 'idx': int }}
        self._reset_tails: Dict[tuple, dict] = {}

    # -------- Path discovery --------
    def _discover_mission_paths(self, robot_types: List[str], base_path: str) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for rt in robot_types:
            rt_dir = os.path.join(base_path, rt)
            if not os.path.isdir(rt_dir):
                self.logger.warning(f"Robot type directory not found: {rt_dir}")
                continue
            missions: Dict[str, str] = {}
            for mission in SUPPORTED_MISSIONS:
                m_dir = os.path.join(rt_dir, mission)
                if os.path.isdir(m_dir):
                    missions[mission] = m_dir
            if missions:
                out[rt] = missions
        return out

    # -------- Parquet loading --------
    def _load_parquet_actions_exact(self, path: str) -> np.ndarray:
        """Load 'action' column and return ndarray(T, A) with strict parity.

        - No scaling, no interpolation, no reordering/permutation.
        - Filters out rows that do not match the most common action dimension
          to avoid shape glitches.
        """
        df = pd.read_parquet(path)
        if 'action' not in df.columns:
            raise ValueError(f"{path} missing required 'action' column")
        actions = df['action'].tolist()
        # Determine most common length to enforce consistency
        lengths = {}
        for idx, a in enumerate(actions):
            n = int(np.asarray(a, dtype=float).size)
            lengths[n] = lengths.get(n, 0) + 1
        if not lengths:
            raise ValueError(f"{path} contains no action data")
        target_dim = max(lengths.items(), key=lambda kv: kv[1])[0]
        filtered: List[np.ndarray] = []
        skipped = 0
        for i, a in enumerate(actions):
            arr = np.asarray(a, dtype=float).ravel()
            if arr.size == target_dim:
                filtered.append(arr)
            else:
                skipped += 1
        if skipped:
            self.logger.warning(f"{path}: skipped {skipped} rows due to dim mismatch; using dim={target_dim}")
        if not filtered:
            raise ValueError(f"{path}: no rows with consistent action dim")
        return np.stack(filtered, axis=0)

    def _try_build_actions_from_obs(self, mission_folder: str, parquet_path: str, expected_dim: int) -> Optional[np.ndarray]:
        """If obs_to_action.json exists in mission folder, build action frames from observation.state.

        obs_to_action.json format:
        {
          "indices": [i0, i1, ..., iA-1]  // indices into observation.state picking joint qpos
        }

        Returns ndarray(T,A) or None on failure.
        """
        try:
            sidecar = os.path.join(mission_folder, "obs_to_action.json")
            use_sidecar = os.path.isfile(sidecar)
            idxs = None
            if use_sidecar:
                spec = json.load(open(sidecar, "r"))
                idxs = spec.get("indices")
                if not isinstance(idxs, list) or not idxs:
                    self.logger.warning(f"Invalid obs_to_action.json (no indices): {sidecar}")
                    use_sidecar = False
            df = pd.read_parquet(parquet_path)
            if 'observation.state' not in df.columns:
                self.logger.warning(f"{parquet_path} has no 'observation.state' column")
                return None
            obs = df['observation.state'].tolist()
            # Determine mapping indices
            if not use_sidecar:
                # Heuristic: take first N qpos as robot joints
                N = int(os.getenv('REPLAY_OBS_FIRST_N', '7'))
                idxs = list(range(N))
            # Build framesbu
            filtered: List[np.ndarray] = []
            for row in obs:
                arr = np.asarray(row, dtype=float).ravel()
                if max(idxs) >= arr.size:
                    continue
                sel = arr[idxs]
                filtered.append(sel)
            if not filtered:
                self.logger.warning(f"{parquet_path}: could not map any observation rows with given indices")
                return None
            out = np.stack(filtered, axis=0)
            # If expected_dim indicates gripper present, try to append from original action column if available
            if expected_dim > out.shape[1]:
                if 'action' in df.columns:
                    acts = df['action'].tolist()
                    app: List[float] = []
                    for a in acts[:out.shape[0]]:
                        av = np.asarray(a, dtype=float).ravel()
                        if av.size >= expected_dim:
                            app.append(float(av[expected_dim-1]))
                        else:
                            app.append(0.0)
                    out = np.hstack([out, np.array(app, dtype=float).reshape(-1, 1)])
                else:
                    # Fallback constant gripper
                    out = np.hstack([out, np.full((out.shape[0], expected_dim - out.shape[1]), 0.0, dtype=float)])
            src = sidecar if use_sidecar else f"first_{len(idxs)}_obs"
            self.logger.info(f"Using observation.state as actions via {src}; dim={out.shape[1]}")
            return out
        except Exception as e:
            self.logger.warning(f"Failed building actions from obs for {parquet_path}: {e}")
            return None

    def _time_resample_stepwise(self, frames: np.ndarray, speed: float) -> np.ndarray:
        """Stepwise time resample to change playback speed without altering values.

        - speed > 1.0: faster → skip frames with integer stride ~speed
        - speed < 1.0: slower → repeat each frame ~1/speed times
        """
        if speed == 1.0 or frames.shape[0] <= 1:
            return frames
        if speed > 1.0:
            stride = max(1, int(round(speed)))
            return frames[::stride]
        # speed < 1.0
        reps = max(1, int(round(1.0 / max(1e-9, speed))))
        return np.repeat(frames, reps, axis=0)

    # -------- Synthetic reset synthesis --------
    def _build_synthetic_reset(self, robot_type: str, mission: str) -> Optional[np.ndarray]:
        """Create a simple reset sequence that holds the initial joint targets for a duration.

        Uses latest saved joint-only state (.npz) from REPLAY_SAVED_STATE_DIR or
        finetuning/saved_robot_states. Maps joint1..joint7 to first 7 action dims and
        sets gripper to 0.0 (open) by default. Duration controlled by RESET_HOLD_SECONDS
        and CONTROL_HZ envs.
        """
        try:
            import glob
            base = os.getenv("MAIN_DIRECTORY") or os.getcwd()
            override = os.getenv("REPLAY_SAVED_STATE_DIR", "").strip()
            search_dirs = [d for d in [override, os.path.join(base, "finetuning", "saved_robot_states")] if d]
            candidates = []
            for d in search_dirs:
                if os.path.isdir(d):
                    candidates.extend(sorted(glob.glob(os.path.join(d, "*.npz")), key=os.path.getmtime))
            if not candidates:
                self.logger.warning("No saved robot state found for synthetic reset; skipping")
                return None
            path = candidates[-1]
            arr = np.load(path, allow_pickle=True)
            if not ("joint_names" in arr and "qpos" in arr):
                self.logger.warning(f"Saved state missing joint_names/qpos: {path}")
                return None
            names = [str(x) for x in arr["joint_names"].tolist()]
            qpos = np.array(arr["qpos"], dtype=float).ravel()
            # Expect Franka Panda joints 'joint1'..'joint7'
            target7 = []
            for i in range(1, 8):
                nm = f"joint{i}"
                try:
                    idx = names.index(nm)
                    target7.append(float(qpos[idx]))
                except ValueError:
                    target7.append(0.0)
            # Append gripper (open=0.0 by default). Adjust if your gripper uses another neutral.
            target = np.array(target7 + [0.0], dtype=float)

            # Determine frames to hold
            try:
                hz = float(os.getenv("CONTROL_HZ", "60"))
            except Exception:
                hz = 60.0
            try:
                seconds = float(os.getenv("RESET_HOLD_SECONDS", "2.0"))
            except Exception:
                seconds = 2.0
            frames = max(1, int(round(hz * seconds)))
            out = np.tile(target.reshape(1, -1), (frames, 1))
            return out
        except Exception as e:
            self.logger.warning(f"Failed to synthesize reset sequence: {e}")
            return None

    # -------- Control flow --------
    def reset(self, robot_id: Optional[str] = None, robot_type: Optional[str] = None, mission: Optional[str] = None):
        if robot_id and robot_type and mission:
            self.robot_mission_state[(robot_id, robot_type, mission)] = {'i_sub': 0, 'i_frame': 0}
        else:
            self.robot_mission_state.clear()

    def next_action(self, robot_id: str, robot_type: str, mission: str) -> List[float]:
        # Reserved auto-reset mission: serve synthesized reset sequence regardless of dataset
        if mission == "__auto_reset__":
            tail_key = (robot_id, robot_type, mission)
            tail = self._reset_tails.get(tail_key)
            if tail is None:
                synth = self._build_synthetic_reset(robot_type, mission)
                if synth is not None and synth.shape[0] > 0:
                    self._reset_tails[tail_key] = {'frames': synth, 'idx': 0}
                    tail = self._reset_tails[tail_key]
            if tail is not None:
                frames = tail['frames']
                idx = tail['idx']
                if idx < frames.shape[0]:
                    out = frames[idx].astype(float).ravel()
                    tail['idx'] = idx + 1
                    return out.tolist()
                # tail finished
                self._reset_tails.pop(tail_key, None)
            return self._zeros_for_mission(robot_type, mission)

        key_state = (robot_id, robot_type, mission)
        if key_state not in self.robot_mission_state:
            self.robot_mission_state[key_state] = {'i_sub': 0, 'i_frame': 0}

        state = self.robot_mission_state[key_state]
        sub_list = self.mission_sub_missions.get((robot_type, mission), [])
        if not sub_list:
            self.logger.error(f"No sub-missions for {robot_type}/{mission}")
            return self._zeros_for_mission(robot_type, mission)

        # Advance across sub-missions as frames are exhausted
        while state['i_sub'] < len(sub_list):
            sm = sub_list[state['i_sub']]
            key = (robot_type, mission, sm)
            arr = self.episode_actions.get(key)
            if arr is None or arr.shape[0] == 0:
                self.logger.warning(f"Empty or missing actions for {key}; skipping")
                state['i_sub'] += 1
                state['i_frame'] = 0
                continue
            if state['i_frame'] >= arr.shape[0]:
                state['i_sub'] += 1
                state['i_frame'] = 0
                continue
            out = arr[state['i_frame']].astype(float).ravel()
            state['i_frame'] += 1
            return out.tolist()

        # Completed mission frames: optionally serve a synthetic reset tail
        tail_key = (robot_id, robot_type, mission)
        tail = self._reset_tails.get(tail_key)
        if tail is None:
            synth = self._build_synthetic_reset(robot_type, mission)
            if synth is not None and synth.shape[0] > 0:
                self._reset_tails[tail_key] = {'frames': synth, 'idx': 0}
                tail = self._reset_tails[tail_key]
        if tail is not None:
            frames = tail['frames']
            idx = tail['idx']
            if idx < frames.shape[0]:
                out = frames[idx].astype(float).ravel()
                tail['idx'] = idx + 1
                return out.tolist()
            # tail finished; drop it
            self._reset_tails.pop(tail_key, None)

        # Completed everything
        if self.loop and sub_list:
            state['i_sub'] = 0
            state['i_frame'] = 0
            return self.next_action(robot_id, robot_type, mission)
        return self._zeros_for_mission(robot_type, mission)

    # -------- Progress --------
    def get_progress(self, robot_id: str, robot_type: str, mission: str) -> float:
        # Auto-reset mission: progress based on reset tail
        if mission == "__auto_reset__":
            tail = self._reset_tails.get((robot_id, robot_type, mission))
            if not tail:
                return 1.0
            frames = tail.get('frames')
            idx = int(tail.get('idx', 0))
            if not isinstance(frames, np.ndarray) or frames.shape[0] == 0:
                return 1.0
            return min(idx / frames.shape[0], 1.0)
        key_state = (robot_id, robot_type, mission)
        state = self.robot_mission_state.get(key_state)
        sub_list = self.mission_sub_missions.get((robot_type, mission), [])
        if not state or not sub_list:
            return 0.0
        total = 0
        done = 0
        for i, sm in enumerate(sub_list):
            key = (robot_type, mission, sm)
            n = self.episode_lengths.get(key, 0)
            total += n
            if i < state['i_sub']:
                done += n
            elif i == state['i_sub']:
                done += min(state['i_frame'], n)
        base = 0.0 if total == 0 else min(done / total, 1.0)
        # If a reset tail is pending, keep progress strictly less than 1.0 to prevent early completion
        if self.is_reset_pending(robot_id, robot_type, mission):
            # Nudge below 1.0 but keep monotonic behavior
            return min(base, 0.999)
        return base

    def is_reset_pending(self, robot_id: str, robot_type: str, mission: str) -> bool:
        tail = self._reset_tails.get((robot_id, robot_type, mission))
        if not tail:
            return False
        frames = tail.get('frames')
        idx = int(tail.get('idx', 0))
        return isinstance(frames, np.ndarray) and idx < frames.shape[0]

    # -------- Utilities --------
    def load_type(self, robot_type: str) -> bool:
        return True

    def _zeros_for_mission(self, robot_type: str, mission: str) -> List[float]:
        # Try to infer from any loaded sub-mission of this mission
        sub_list = self.mission_sub_missions.get((robot_type, mission), [])
        for sm in sub_list:
            arr = self.episode_actions.get((robot_type, mission, sm))
            if arr is not None and arr.size > 0:
                return [0.0] * int(arr.shape[1])
        # Fallback: any other loaded episode
        for key, arr in self.episode_actions.items():
            if isinstance(key, tuple) and len(key) == 3 and arr is not None and arr.size > 0:
                return [0.0] * int(arr.shape[1])
        return [0.0]
