#!/usr/bin/env python3

import subprocess
import socket
import pickle
import struct
import time
import threading
import signal
import sys
from control_panel.mission_manager import MissionManager
from mujoco_folder.packet_example import Packet
from logger_config import get_logger
from dotenv import load_dotenv
import os
from mujoco_folder.mujoco_server import MujocoClient
from pathlib import Path


load_dotenv()

class MainOrchestrator:
    def __init__(self):
        self.logger = get_logger('MainOrchestrator')
        
        self.mujoco_process = None
        self.cli_process = None
        self.mujoco_client = MujocoClient()
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, _signum, _frame):
        self.logger.info("Shutdown signal received")
        self.shutdown()
        
    def start_mujoco_server(self):
        self.logger.info("Starting MuJoCo server...")
        project_root = Path(__file__).resolve().parent  # .../autonomous
        self.mujoco_process = subprocess.Popen([
            sys.executable,
            "-m", "mujoco_folder.mujoco_server"
        ], cwd=str(project_root))


    def start_queue_server(self):
        self.logger.info("Starting Queue server...")
        project_root = Path(__file__).resolve().parent  # .../autonomous
        self.queue_process = subprocess.Popen(
            [sys.executable, "-m", "control_panel.robot_queue_locks"],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": str(project_root)}
        )
        time.sleep(2)
    
    def start_logging_server(self):
        print("Starting Logging server...")
        project_root = Path(__file__).resolve().parent
        self.logging_process = subprocess.Popen([
            sys.executable,
            "-m", "logger_config"
        ], cwd=str(project_root))
        time.sleep(1)  # Give server time to start



    def start_cli(self):
        self.logger.info("Starting CLI...")
        repo_root = Path(__file__).resolve().parent  # .../autonomous

        self.brain_process = subprocess.Popen(
            [sys.executable, "-m", "brain.run_cli"],
            cwd=str(repo_root),
            env={**os.environ, "PYTHONPATH": str(repo_root)}
        )
        
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

    def connect_to_brain(self):
        self.logger.info("Connecting to brain...")
        # Placeholder for brain connection logic
        # This could be a separate client connection or API call
        pass

                    
    def inference_loop(self, robot_id, mission_manager, tries=3):
        self.logger.info("Starting inference loop for robot: %s", robot_id)

        # Step 0: Start an empty packet with robot id
        packet = Packet(robot_id=robot_id)
        while packet.mission is None:
            #self.logger.debug(f"Waiting for mission for robot {robot_id}...")
            result = mission_manager.get_next_mission(robot_id)
            if result['robot_id'] is not None and result['robot_id'] != robot_id:
                self.logger.error(f'Robot ID in mission pair is not the one needed {result}')
                raise
            packet.mission = result['mission']

            time.sleep(0.3)
        
        self.logger.info(f"Robot {robot_id} assigned mission: {packet.mission}")

        while self.running:
            try:
                # Step 1: Send to Queue, Dequeue from Queue, assign to robot
                packet = self.mujoco_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Received robot state")
                if packet is None:
                    self.logger.warning(f"Robot {robot_id} state is None, skipping...")
                    time.sleep(0.1)
                    continue
                    
                # Step 3: Send to mission analyzer, get mission state
                packet = mission_manager.manage_mission(packet)

                if packet.mission is None:
                    self.logger.info(f"Robot {robot_id} has no mission, reset for next mission")
                    self.inference_loop(robot_id, mission_manager)
                
                # TODO: Step 5: Send to policy, get action
                # This is where the policy inference will happen
                # For now, using dummy action
                dummy_action = [0.0] * 7  # 7-DOF robot arm
  
                
                # Step 6: Send to mujoco server map actions loop
                packet.action = dummy_action
                packet = self.mujoco_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id}Action sent, result: {packet.action}")

                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in inference loop: {e}")
                time.sleep(1)
                self.logger.info(f"Retrying inference loop for robot {robot_id}, attempt {tries}")
                tries -= 1
                if tries <= 0:
                    self.logger.error(f"Max retries reached for robot {robot_id}, exiting loop")
                    raise
                self.inference_loop(robot_id, mission_manager, tries)
                
    def run(self):
        try:
            self.start_logging_server()  # Start this first
            self.start_mujoco_server()
            self.start_queue_server()
            
            time.sleep(2)
            self.connect_to_mujoco()
            self.connect_to_brain()
            self.robot_list = self.mujoco_client.recv_robot_list()
            if not self.robot_list:
                self.logger.error("No robots found in the robot list")
                return
                
            self.logger.info(f"Robot list received: {self.robot_list}")
            MissionManager.set_robot_list(self.robot_list)
            

            for robot_id in self.robot_list:
                self.logger.info(f"Robot {robot_id} is ready for missions")
                manager = MissionManager()
                inference_thread = threading.Thread(target=self.inference_loop, daemon=True, args=(robot_id, manager))
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
                
        if self.queue_process:
            self.queue_process.terminate()
            try:
                self.queue_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.queue_process.kill()
        
        if self.logging_process:
            self.logging_process.terminate()
            try:
                self.logging_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logging_process.kill()
        print("All processes terminated, exiting...")
        sys.exit(0)

if __name__ == '__main__':
    orchestrator = MainOrchestrator()
    orchestrator.run()