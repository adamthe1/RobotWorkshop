#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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

    def __init__(self, parquet_path: str, loop: bool = True, slicing: int = 4):
        p = Path(parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.df = pd.read_parquet(p)
        if 'action' not in self.df.columns:
            raise ValueError("Parquet missing 'action' column")
        # Normalize actions to np.ndarray of shape (T, A)
        actions = self.df['action'].tolist()
        if slicing > 1:
            actions = actions[::slicing]
        try:
            self.actions = np.asarray(actions, dtype=object)
        except Exception:
            # Fall back to list-of-lists
            self.actions = actions
        self.T = len(self.actions)
        self.loop = loop
        # Use single cursor for robot ID agnostic replay
        self.cursor = 0
        print(f"[EpisodeActionMapper] Loaded {self.T} actions, robot ID agnostic mode")

    def reset(self, robot_id: Optional[str] = None):
        """Reset cursor to beginning - robot_id parameter ignored for compatibility"""
        self.cursor = 0

    def next_action(self, robot_id: str, joint_names: Optional[List[str]] = None,
                    qpos: Optional[List[float]] = None, qvel: Optional[List[float]] = None) -> List[float]:
        """
        Return the next action - robot_id parameter ignored for compatibility.
        
        For now this replays sequentially. It does not yet align by state/timestamp,
        but that can be added (e.g., nearest-neighbor on state or by stored timestamps).
        """
        idx = self.cursor
        if idx >= self.T:
            if self.loop and self.T > 0:
                idx = 0
                self.cursor = 0
            else:
                # No more actions; return zeros with appropriate length if possible
                length = len(self.actions[0]) if self.T > 0 else 0
                return [0.0] * length

        act = self.actions[idx]
        # Convert to flat list of floats
        if isinstance(act, np.ndarray):
            act_out = act.astype(float).ravel().tolist()
        else:
            act_out = [float(x) for x in act]

        # Advance cursor
        self.cursor = idx + 1
        return act_out
    
    def get_progress(self) -> float:
        """Return progress through the episode as a float in [0, 1]."""
        if self.T == 0:
            return 1.0
        return min(1.0, self.cursor / self.T)

