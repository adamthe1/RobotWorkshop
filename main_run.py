#!/usr/bin/env python3

import time
import threading
import signal
import sys
from control_panel.mission_manager import MissionManager
from mujoco_folder.packet_example import Packet
from logger_config import get_logger
from dotenv import load_dotenv
import os
from mujoco_folder.mujoco_server_merge import MujocoClient
from brain.brain_server import BrainClient
from control_panel.robot_queue_locks import QueueClient
from pathlib import Path
from mujoco_folder.server_manager import ServerManager

load_dotenv()

class MainOrchestrator:
    def __init__(self):
        self.logger = get_logger('MainOrchestrator')
        
        self.server_manager = ServerManager()
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, _signum, _frame):
        self.logger.info("Shutdown signal received")
        self.shutdown()
        
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
                    self.logger.error("Failed to connect to MuJoCo server after max retries")
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
                    self.logger.warning(f"Brain connection attempt {i+1} failed, retrying...")
                    time.sleep(1)
                else:
                    self.logger.error("Failed to connect to Brain server after max retries")
                    raise
    
    def clean_packet(self, packet):
        packet.action = None
        packet.qpos = None
        packet.qvel = None
        packet.joint_names = None
        packet.wall_camera = None
        packet.wrist_camera = None
        return packet

    def inference_loop(self, robot_id, clients, tries=3):
        self.logger.info("Starting inference loop for robot: %s", robot_id)
        mission_manager = clients['manager']
        mujoco_client = clients['mujoco']
        brain_client = clients['brain']
        
        # Step 0: Start an empty packet with robot id
        packet = Packet(robot_id=robot_id)
        while packet.mission is None:
            result = mission_manager.get_next_mission(robot_id)
            if result['robot_id'] is not None and result['robot_id'] != robot_id:
                self.logger.error(f'Robot ID in mission pair is not the one needed {result}')
                raise Exception("Robot ID mismatch")
            packet.mission = result['mission']
            mission = result['mission']
            time.sleep(0.3)

        self.logger.info(f"Robot {robot_id} assigned mission: {packet}")
        print(f"\nHey, {robot_id} is preparing your drink, please come to it's bar\n")

        while self.running:
            try:
                packet = self.clean_packet(packet)
                # Step 1: Send to Queue, Dequeue from Queue, assign to robot
                packet = mujoco_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Received robot state: {packet.qpos}")

                
                if packet is None:
                    self.logger.error(f"Packet is None from Mujoco for robot {robot_id}")
                    raise Exception("Did not return packet from Mujoco")

                # Step 3: Send to mission analyzer, get mission state
                packet = mission_manager.manage_mission(packet)

                if packet.mission is None:
                    self.logger.info(f"Robot {robot_id} has no mission, resetting for next mission")
                    self.inference_loop(robot_id, clients)
                
                # Step 4: Send to Brain, get action
                packet = brain_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Received action from Brain {packet}")

                packet = mujoco_client.send_and_recv(packet)
                self.logger.debug(f"{robot_id} Action sent, result: {packet}")

                # Align loop with simulation/control rate for precise replay
                # Default to 60 Hz if not set
                try:
                    hz = float(os.getenv("CONTROL_HZ", "60"))
                except Exception:
                    hz = 60.0
                time.sleep(max(0.0, 1.0/ hz))
                
            except Exception as e:
                self.logger.error(f"Error in inference loop: {e}")
                time.sleep(1)
                self.logger.info(f"Retrying inference loop for robot {robot_id}, attempt {tries}")
                tries -= 1
                if tries <= 0:
                    self.logger.error(f"Max retries reached for robot {robot_id}, exiting loop")
                    self.shutdown()
                mission_manager.reset_robot_and_mission(robot_id, mission)
                self.inference_loop(robot_id, clients, tries)
                
                
    def run(self):
        try:
            # Start all servers using server manager
            self.logger.info("Starting all servers...")
            self.server_manager.start_all_servers()

            # Wait for servers to fully start
            time.sleep(3)

            # Get robot list from MuJoCo
            self.logger.info("Getting robot list...")
            self.robot_list, self.robot_dict = MujocoClient().recv_robot_list_and_dict()

            
            if not self.robot_list:
                self.logger.error("No robots found in the robot list")
                return

            self.logger.info(f"Robot list received: {self.robot_list}")
            MissionManager.set_robot_list(self.robot_list)
            BrainClient.set_robot_dict(self.robot_dict)
            QueueClient.set_robot_dict(self.robot_dict)


            # Start inference loops for each robot
            for robot_id in self.robot_list:
                self.logger.info(f"Setting up robot {robot_id} for missions")
                manager = MissionManager()
                mujoco_client = MujocoClient()
                brain_client = BrainClient()   
                
                self.connect_to_mujoco(mujoco_client)
                self.connect_to_brain(brain_client)
                
                clients = {"manager": manager, "mujoco": mujoco_client, "brain": brain_client}
                inference_thread = threading.Thread(
                    target=self.inference_loop, 
                    daemon=True, 
                    args=(robot_id, clients)
                )
                inference_thread.start()
                self.logger.info(f"Started inference thread for robot {robot_id}")

            # Start CLI
            self.server_manager.start_cli()
            
            # Monitor servers
            while self.running:
                status = self.server_manager.get_server_status()
                terminated_servers = [name for name, stat in status.items() if "Terminated" in stat]
                
                if terminated_servers:
                    self.logger.error(f"Servers terminated unexpectedly: {terminated_servers}")
                    break

                time.sleep(3)  # Check every 3 seconds

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
            
    def shutdown(self):
        self.running = False
        self.logger.info("Shutting down MainOrchestrator...")
        
        # Use server manager to cleanly shutdown all servers
        self.server_manager.kill_all_servers()
        
        time.sleep(2)
        self.logger.info("All processes terminated, exiting...")
        sys.exit(0)

if __name__ == '__main__':
    orchestrator = MainOrchestrator()
    orchestrator.run()
