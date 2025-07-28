#!/usr/bin/env python3

import subprocess
import socket
import pickle
import struct
import time
import threading
import signal
import sys
from control_panel.robot_queue import RobotQueue
from control_panel.mission_manager import MissionManager
from packet_example import Packet
from logger_config import get_logger

class MujocoClient:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        
    def _send_packet(self, packet):
        data = pickle.dumps(packet)
        self.socket.sendall(struct.pack('!I', len(data)) + data)
        
    def _recv_packet(self):
        size_data = self.socket.recv(4)
        if not size_data:
            return None
        size = struct.unpack('!I', size_data)[0]
        buf = b''
        while len(buf) < size:
            buf += self.socket.recv(size - len(buf))
        return pickle.loads(buf)
        
    def get_robot_state(self, robot_id):
        packet = Packet(robot_id=robot_id)
        self._send_packet(packet)
        return self._recv_packet()
        
    def send_action(self, robot_id, action):
        packet = Packet(robot_id=robot_id, action=action)
        self._send_packet(packet)
        return self._recv_packet()
        
    def close(self):
        if self.socket:
            self.socket.close()

class MainOrchestrator:
    def __init__(self):
        self.logger = get_logger('MainOrchestrator')
        
        self.mujoco_process = None
        self.cli_process = None
        self.mujoco_client = MujocoClient()
        self.robot_queue = RobotQueue(['robot_1'])
        self.mission_manager = MissionManager()
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, _signum, _frame):
        self.logger.info("Shutdown signal received")
        self.shutdown()
        
    def start_mujoco_server(self):
        self.logger.info("Starting MuJoCo server...")
        self.mujoco_process = subprocess.Popen([
            sys.executable, 'mujoco_server.py'
        ])
        time.sleep(2)
        
    def start_cli(self):
        self.logger.info("Starting CLI...")
        self.cli_process = subprocess.Popen([
            sys.executable, 'run.py'
        ])
        
    def connect_to_mujoco(self):
        self.logger.info("Connecting to MuJoCo server...")
        max_retries = 10
        for i in range(max_retries):
            try:
                self.mujoco_client.connect()
                self.logger.info("Connected to MuJoCo server")
                return
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    self.logger.warning(f"Connection attempt {i+1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise
                    
    def inference_loop(self):
        self.logger.info("Starting inference loop...")
        robot_id = 'robot_1'
        current_mission = None
        current_submission = None
        
        while self.running:
            try:
                # Step 1: Send to Queue, Dequeue from Queue, assign to robot
                if current_mission is None:
                    mission = self.robot_queue.dequeue_mission(robot_id)
                    if mission:
                        current_mission = mission
                        current_submission = None
                        self.logger.info(f"Assigned mission '{mission}' to {robot_id}")
                
                if current_mission is None:
                    time.sleep(0.1)
                    continue
                    
                # Step 2: Send empty packet with robot id to robot, get robot state
                robot_state = self.mujoco_client.get_robot_state(robot_id)
                if robot_state is None:
                    continue
                    
                # Add mission context to robot state
                robot_state.current_mission = current_mission
                robot_state.current_submission = current_submission
                
                # Step 3: Send to mission analyzer, get mission state
                mission_result = self.mission_manager.manage_mission({
                    'current_mission': current_mission,
                    'current_submission': current_submission,
                    'robot_state': robot_state
                })
                
                # Step 4: Send to submission get submission
                if mission_result is None:
                    current_mission = None
                    current_submission = None
                    continue
                elif mission_result == "reset before new mission":
                    current_mission = None
                    current_submission = None
                    continue
                else:
                    current_submission = mission_result
                
                # TODO: Step 5: Send to policy, get action
                # This is where the policy inference will happen
                # For now, using dummy action
                dummy_action = [0.0] * 7  # 7-DOF robot arm
                self.logger.info(f"TODO: Policy inference for submission '{current_submission}'")
                self.logger.debug(f"TODO: Using dummy action: {dummy_action}")
                
                # Step 6: Send to mujoco server map actions loop
                action_result = self.mujoco_client.send_action(robot_id, dummy_action)
                self.logger.debug(f"Action sent, result: {action_result}")
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in inference loop: {e}")
                time.sleep(1)
                
    def run(self):
        try:
            self.start_mujoco_server()
            self.connect_to_mujoco()
            
            inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
            inference_thread.start()
            
            self.start_cli()
            
            while self.running:
                if self.mujoco_process and self.mujoco_process.poll() is not None:
                    self.logger.warning("MuJoCo server process terminated")
                    break
                if self.cli_process and self.cli_process.poll() is not None:
                    self.logger.warning("CLI process terminated")
                    break
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.shutdown()
            
    def shutdown(self):
        self.running = False
        self.logger.info("Shutting down all processes...")
        
        if self.mujoco_client:
            self.mujoco_client.close()
            
        if self.cli_process:
            self.cli_process.terminate()
            try:
                self.cli_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.cli_process.kill()
                
        if self.mujoco_process:
            self.mujoco_process.terminate()
            try:
                self.mujoco_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mujoco_process.kill()
                
        self.logger.info("Shutdown complete")
        sys.exit(0)

if __name__ == '__main__':
    orchestrator = MainOrchestrator()
    orchestrator.run()