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
from brain.brain_server import BrainClient
from pathlib import Path


load_dotenv()

class MainOrchestrator:
    def __init__(self):
        self.logger = get_logger('MainOrchestrator')
        
        self.mujoco_process = None
        self.cli_process = None
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
        time.sleep(1)  # Give server time to start


    def start_queue_server(self):
        self.logger.info("Starting Queue server...")
        project_root = Path(__file__).resolve().parent  # .../autonomous
        self.queue_process = subprocess.Popen(
            [sys.executable, "-m", "control_panel.robot_queue_locks"],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": str(project_root)}
        )
        time.sleep(1)
    
    def start_logging_server(self):
        print("Starting Logging server...")
        project_root = Path(__file__).resolve().parent
        self.logging_process = subprocess.Popen([
            sys.executable,
            "-m", "logger_config"
        ], cwd=str(project_root))
        time.sleep(1)  # Give server time to start

    def start_brain_server(self):
        self.logger.info("Starting Brain server...")
        project_root = Path(__file__).resolve().parent
        self.brain_process = subprocess.Popen(
            [sys.executable, "-m", "brain.brain_server"],
            cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)}
        )
        time.sleep(1)  # Give server time to start

    def start_servers(self):
        self.start_logging_server()
        self.start_mujoco_server()
        self.start_queue_server()
        self.start_brain_server()


    def start_cli(self):
        self.logger.info("Starting CLI...")
        repo_root = Path(__file__).resolve().parent  # .../autonomous

        self.brain_process = subprocess.Popen(
            [sys.executable, "-m", "brain.run_cli"],
            cwd=str(repo_root),
            env={**os.environ, "PYTHONPATH": str(repo_root)}
        )
        
    def connect_to_mujoco(self, mujoco_client):
        max_retries = 10
        for i in range(max_retries):
            try:
                mujoco_client.connect()
                self.logger.info("Connected to MuJoCo server")
                return
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    self.logger.warning(f"Connection attempt {i+1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise

    def connect_to_brain(self, brain_client):
        self.logger.info("Connecting to brain...")
        max_retries = 10
        for i in range(max_retries):
            try:
                brain_client.connect()
                self.logger.info("Connected to Brain server")
                return
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    self.logger.warning(f"Connection attempt {i+1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise

    def inference_loop(self, robot_id, clients, tries=3):
        self.logger.info("Starting inference loop for robot: %s", robot_id)
        mission_manager = clients['manager']
        mujoco_client = clients['mujoco']
        brain_client = clients['brain']
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
                packet = mujoco_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Received robot state")
                if packet is None:
                    self.logger.warning(f"Robot {robot_id} state is None, skipping...")
                    time.sleep(0.1)
                    continue
                    
                # Step 3: Send to mission analyzer, get mission state
                packet = mission_manager.manage_mission(packet)

                if packet.mission is None:
                    self.logger.info(f"Robot {robot_id} has no mission, reset for next mission")
                    self.inference_loop(robot_id, clients)
                
                # Step 4: Send to Brain, get action
                packet = brain_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Received action from Brain")

                packet = mujoco_client.send_and_recv(packet)
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
                self.inference_loop(robot_id, clients, tries)
                
    def run(self):
        try:
            self.start_servers()

            time.sleep(2)

            self.robot_list = MujocoClient().recv_robot_list()
            if not self.robot_list:
                self.logger.error("No robots found in the robot list")
                return

            self.logger.info(f"Robot list received: {self.robot_list}")
            MissionManager.set_robot_list(self.robot_list)
            

            for robot_id in self.robot_list:
                self.logger.info(f"Robot {robot_id} is ready for missions")
                manager = MissionManager()
                mujoco_client = MujocoClient()
                brain_client = BrainClient()   
                self.connect_to_mujoco(mujoco_client)
                self.connect_to_brain(brain_client)
                clients = {"manager": manager, "mujoco": mujoco_client, "brain": brain_client}
                inference_thread = threading.Thread(target=self.inference_loop, daemon=True, args=(robot_id, clients))
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

        if self.brain_process:
            self.brain_process.terminate()
            try:
                self.brain_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.brain_process.kill()
        
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