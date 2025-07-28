import threading
import queue
from typing import Dict, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import get_logger

class RobotQueue:
    """
    Holds locks and manages mission queueing and allocation for robots (SOLID-compliant)
    """
    def __init__(self, robot_ids: Optional[List[str]] = None):
        self.logger = get_logger('RobotQueue')
        
        # Store locks for each robot to prevent concurrent access
        self.locks: Dict[str, threading.Lock] = {}
        # Store mission queues for each robot
        self.queues: Dict[str, queue.Queue] = {}
        
        # Initialize robots if provided during construction
        if robot_ids:
            for robot_id in robot_ids:
                self.locks[robot_id] = threading.Lock()
                self.queues[robot_id] = queue.Queue()
                self.logger.info(f"Initialized robot queue for {robot_id}")
    
    def register_robot(self, robot_id: str):
        """Register a new robot with lock and queue if not already registered"""
        if robot_id not in self.locks:
            self.locks[robot_id] = threading.Lock()
            self.queues[robot_id] = queue.Queue()
            self.logger.info(f"Registered new robot: {robot_id}")
    
    def acquire(self, robot_id: str, blocking=True, timeout=None) -> bool:
        """Acquire lock for robot, registering if needed"""
        self.register_robot(robot_id)
        acquired = self.locks[robot_id].acquire(blocking, timeout)
        if acquired:
            self.logger.debug(f"Acquired lock for {robot_id}")
        else:
            self.logger.warning(f"Failed to acquire lock for {robot_id}")
        return acquired
    
    def release(self, robot_id: str):
        """Release lock for robot if it exists"""
        if robot_id in self.locks:
            self.locks[robot_id].release()
            self.logger.debug(f"Released lock for {robot_id}")
    
    def enqueue_mission(self, robot_id: str, mission):
        """Add mission to robot's queue, registering robot if needed"""
        self.register_robot(robot_id)
        self.queues[robot_id].put(mission)
        queue_size = self.queues[robot_id].qsize()
        self.logger.info(f"Enqueued mission '{mission}' for {robot_id} (queue size: {queue_size})")
    
    def dequeue_mission(self, robot_id: str):
        """Remove and return next mission from robot's queue, or None if empty"""
        self.register_robot(robot_id)
        if not self.queues[robot_id].empty():
            mission = self.queues[robot_id].get()
            queue_size = self.queues[robot_id].qsize()
            self.logger.info(f"Dequeued mission '{mission}' from {robot_id} (remaining in queue: {queue_size})")
            return mission
        return None
    
    def robot_available(self, robot_id: str):
        """Check if robot is available (not locked). 
        Note: This method requires robot to be registered first."""
        if robot_id not in self.locks:
            # Inconsistency: this will raise KeyError if robot not registered
            # Should call register_robot first for consistency
            self.register_robot(robot_id)
        available = not self.locks[robot_id].locked()
        self.logger.debug(f"Robot {robot_id} availability: {available}")
        return available
