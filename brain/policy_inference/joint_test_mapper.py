#!/usr/bin/env python3
"""
Joint Test Action Mapper - generates systematic joint test actions.
Similar interface to EpisodeActionMapper but generates test patterns instead of replaying.
"""

import math
from typing import Dict, List, Optional
import numpy as np


class JointTestMapper:
    """
    Generates systematic joint test actions to verify the action pipeline.
    
    Test patterns include:
    - Individual joint movements (positive/negative)
    - All joints extreme movements
    - Sine wave patterns
    - Alternating patterns
    - Random combinations
    
    Interface matches EpisodeActionMapper for drop-in replacement.
    """

    def __init__(self, types):
        self.types = types
        self.num_joints = 7 # Default, can be updated per robot
        self.test_amplitude = 1.5
        self.loop = True
            
        # Test parameters
        self.sine_frequency = 0.1
        self.steps_per_joint = 20
        self.steps_per_phase = 50
        
        # Test phases
        self.phases = [
            "individual_joints_positive",
            "individual_joints_negative", 
        ]
        
        # Maintain independent cursors per robot
        self.cursors: Dict[str, int] = {}
        self.robot_phases: Dict[str, int] = {}

    def reset(self, robot_id: Optional[str] = None):
        """Reset cursor for specific robot or all robots"""
        if robot_id is None:
            self.cursors = {}
            self.robot_phases = {}
        else:
            self.cursors.pop(robot_id, None)
            self.robot_phases.pop(robot_id, None)

    def next_action(self, robot_id: str, joint_names: Optional[List[str]] = None,
                    qpos: Optional[List[float]] = None, qvel: Optional[List[float]] = None) -> List[float]:
        """
        Generate the next test action for the given robot.
        
        Args:
            robot_id: Robot identifier
            joint_names: List of joint names (used to determine num_joints)
            qpos: Current joint positions (not used for generation but kept for interface)
            qvel: Current joint velocities (not used for generation but kept for interface)
            
        Returns:
            List of action values for each joint
        """
        # Update number of joints if we have joint_names
        if joint_names is not None:
            self.num_joints = len(joint_names)
        elif qpos is not None:
            self.num_joints = len(qpos)
        
        # Get current step for this robot
        step = self.cursors.get(robot_id, 0)
        
        # Determine current phase
        phase_idx = (step // self.steps_per_phase) % len(self.phases)
        current_phase = self.phases[phase_idx]
        step_in_phase = step % self.steps_per_phase
        
        # Generate action based on current phase
        if current_phase == "individual_joints_positive":
            action = self._test_individual_joints_positive(step_in_phase)
        elif current_phase == "individual_joints_negative":
            action = self._test_individual_joints_negative(step_in_phase)
        elif current_phase == "all_joints_extreme_positive":
            action = [self.test_amplitude] * self.num_joints
        elif current_phase == "all_joints_extreme_negative":
            action = [-self.test_amplitude] * self.num_joints
        elif current_phase == "sine_wave_all_joints":
            action = self._test_sine_wave(step)
        elif current_phase == "alternating_pattern":
            action = self._test_alternating_pattern(step)
        elif current_phase == "random_combinations":
            action = self._test_random_combinations(step)
        elif current_phase == "return_to_zero":
            action = [0.0] * self.num_joints
        else:
            action = [0.0] * self.num_joints
        
        # Advance cursor
        self.cursors[robot_id] = step + 1
        
        # If we've completed all phases and loop is disabled, return zeros
        if not self.loop and phase_idx >= len(self.phases) - 1 and step_in_phase >= self.steps_per_phase - 1:
            return [0.0] * self.num_joints
            
        return action

    def _test_individual_joints_positive(self, step_in_phase: int) -> List[float]:
        """Test one joint at a time with positive values"""
        action = [0.0] * self.num_joints
        joint_idx = (step_in_phase // 5) % self.num_joints  # Change joint every 5 steps
        action[joint_idx] = self.test_amplitude
        return action

    def _test_individual_joints_negative(self, step_in_phase: int) -> List[float]:
        """Test one joint at a time with negative values"""
        action = [0.0] * self.num_joints
        joint_idx = (step_in_phase // 5) % self.num_joints  # Change joint every 5 steps
        action[joint_idx] = -self.test_amplitude
        return action

    def _test_sine_wave(self, step: int) -> List[float]:
        """Generate sine wave pattern for all joints with different phases"""
        action = []
        for i in range(self.num_joints):
            # Each joint has a different phase offset
            phase_offset = (i * 2 * math.pi) / self.num_joints
            value = self.test_amplitude * math.sin(step * self.sine_frequency + phase_offset)
            action.append(value)
        return action

    def _test_alternating_pattern(self, step: int) -> List[float]:
        """Alternating positive/negative pattern across joints"""
        action = []
        for i in range(self.num_joints):
            # Alternate based on joint index and test step
            if (i + step) % 2 == 0:
                action.append(self.test_amplitude)
            else:
                action.append(-self.test_amplitude)
        return action

    def _test_random_combinations(self, step: int) -> List[float]:
        """Random combinations within safe limits"""
        np.random.seed(step)  # Reproducible randomness
        action = np.random.uniform(-self.test_amplitude, self.test_amplitude, self.num_joints).tolist()
        return action

    def get_current_phase(self, robot_id: str) -> str:
        """Get current test phase for a robot"""
        step = self.cursors.get(robot_id, 0)
        phase_idx = (step // self.steps_per_phase) % len(self.phases)
        return self.phases[phase_idx]

    def get_progress(self, robot_id: str) -> Dict[str, int]:
        """Get progress information for a robot"""
        step = self.cursors.get(robot_id, 0)
        phase_idx = (step // self.steps_per_phase) % len(self.phases)
        step_in_phase = step % self.steps_per_phase
        
        return {
            "total_step": step,
            "phase_index": phase_idx,
            "step_in_phase": step_in_phase,
            "total_phases": len(self.phases),
            "current_phase": self.phases[phase_idx]
        }