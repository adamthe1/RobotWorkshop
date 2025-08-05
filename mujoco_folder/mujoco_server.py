#!/usr/bin/env python3
# MuJoCo Server Script (dummy state stubs)
# This script initializes MuJoCo, runs physics simulation, and communicates
# with a client via TCP sockets. Non-server functionality is stubbed.

import os
import sys
import socket
import pickle
import struct
import time
import threading
import numpy as np
from pathlib import Path
from .packet_example import Packet, RobotListPacket

import mujoco
import mujoco.viewer
from logger_config import get_logger

from dotenv import load_dotenv
import os

load_dotenv()

class MuJoCoServer:
    def __init__(self, xml_path, host=os.getenv("MUJOCO_HOST", "localhost"), port=int(os.getenv("MUJOCO_PORT", 5555)),
                 render_rgb=True, rgb_width=256, rgb_height=256, no_viewer=False):
        """
        Initialize MuJoCo server to run physics simulation and communicate
        with a client via TCP sockets.
        Non-server internals are stubbed with dummy values.
        """
        self.host = host
        self.port = port
        self.running = True
        self.locker = threading.Lock()
        self.server_socket = None
        self.logger = get_logger('MujocoServer')

        # Load MuJoCo model (real)
        self.logger.info(f"Loading MuJoCo model from: {xml_path}")
        model_path = Path(xml_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data  = mujoco.MjData(self.model)
        self.logger.info("Model and data initialized successfully")

        # Set initial pose (real)
        self.data.qpos[2] = 1.0
        mujoco.mj_forward(self.model, self.data)

        # Viewer settings
        self.render_rgb = render_rgb
        self.rgb_width  = rgb_width
        self.rgb_height = rgb_height
        self.viewer     = None

        self.no_viewer = no_viewer

        # Networking setup
        self.setup_socket()
        self._setup_joint_mappings()



    def setup_socket(self):
        """Set up the TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        



    def _setup_joint_mappings(self):
        """Create dummy joint mapping for stubs"""
        # Stub: one dummy joint
        self.required_joint_names = ['dummy_joint']
        self.required_joint_ids   = [0]



    def start_viewer(self):
        """Launch MuJoCo passive viewer in background thread"""
        try:
            self.logger.info("Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(0.5)
            self.logger.info("Viewer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize viewer: {e}")
            self.viewer = None

    def fill_robot_list(self, packet):
        """
        Fill the packet with a dummy robot list.
        This is a stub method that simulates receiving a list of robots.
        """
        packet.robot_list = self.get_robot_list()
        return packet

    def get_robot_list(self):
        """Return a list of robot IDs (stubbed)"""
        return ['robot_1', 'robot_2', 'robot_3' , 'robot_4', 'robot_5', 'robot_6']

    def step_once(self):
        """Advance the physics by one timestep"""
        with self.locker:
            mujoco.mj_step(self.model, self.data)

    def update_viewer(self):
        """Update the MuJoCo viewer if it's running."""
        if self.viewer is not None and self.viewer.is_running():
            with self.locker:
                self.viewer.sync()


    def get_joint_state_by_robot_id(self, robot_id):
        """
        Stub: Return dummy joint state arrays matching mapping length
        """
        n = len(self.required_joint_names)
        qpos = np.zeros(n)
        qvel = np.zeros(n)
        return {
            'qpos':        qpos,
            'qvel':        qvel,
            'joint_names': self.required_joint_names
        }

    def get_camera_images_by_robot_id(self, robot_id):
        """
        Stub: Return one dummy black image
        """
        img = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
        return [img]

    def apply_commands(self, packet):
        """
        Stub: Read 'action' from packet and ignore
        """
        action = packet.action
        if action is None:
            self.logger.warning("No action provided in packet, skipping apply_commands")
            return packet
        packet.action = None  # Clear action after applying
        self.logger.debug(f"Applying action for robot {packet.robot_id}: {action}")
        return packet  # Return the packet unchanged for now

    def fill_packet(self, packet):
        """
        Enrich packet with dummy state and return it
        """
        robot_id = packet.robot_id
        with self.locker:
            joint_state   = self.get_joint_state_by_robot_id(robot_id)
            camera_images = self.get_camera_images_by_robot_id(robot_id)


        packet.qpos        = joint_state['qpos']
        packet.qvel        = joint_state['qvel']
        packet.joint_names = joint_state['joint_names']
        # choose the appropriate camera field from your dataclass:
        packet.wall_camera  = camera_images     # or packet.wrist_camera
        packet.time        = time.time()

        return packet

    def _recv_packet(self, client_socket):  # Add socket parameter
        """Receive a length‑prefixed pickled packet"""
        size_data = client_socket.recv(4)  # Use parameter instead of self.client_socket
        if not size_data:
            return None
        size = struct.unpack('!I', size_data)[0]
        buf = b''
        while len(buf) < size:
            buf += client_socket.recv(size - len(buf))  # Use parameter
        return pickle.loads(buf)

    def _send_packet(self, packet, client_socket):  # Add socket parameter
        """Send a length‑prefixed pickled packet"""
        data = pickle.dumps(packet)
        client_socket.sendall(struct.pack('!I', len(data)) + data)  # Use parameter

    def handle_client(self, client_socket, addr):
        """
        Handle incoming RPCs: 'action' → apply+step, else → return state
        """
        # Remove this line: self.client_socket = client_socket
        self.logger.info(f"Client connected: {addr}")
        try:
            while True:
                pkt = self._recv_packet(client_socket)  # Pass socket
                if isinstance(pkt, RobotListPacket):
                    self.logger.debug("Received request for robot list")
                    reply = self.fill_robot_list(pkt)
                    self.logger.debug(f"Sending robot list: {reply.robot_list}")
                elif pkt is None:
                    break
                elif pkt.action is not None:
                    reply = self.apply_commands(pkt)
                    self.step_once()
                else:
                    reply = self.fill_packet(pkt)
                self._send_packet(reply, client_socket)  # Pass socket
        except Exception as e:
            self.logger.error(f"Error in client loop: {e}")
        finally:
            client_socket.close()  # Close this specific client
            self.logger.info(f"Client {addr} disconnected")

    def simulation_thread(self, control_hz=60):
        dt = 1.0 / control_hz
        next_time = time.time()
        while self.running:
            # 1) apply whatever the last action was   #TODO
            
            # 2) step the muJoCo sim
            with self.locker:
                mujoco.mj_step(self.model, self.data)

            # 3) redraw viewer (passive or active)
            if self.no_viewer is False:
                self.update_viewer()

            # 4) sleep until next frame
            next_time += dt
            time.sleep(max(0, next_time - time.time()))

    def run(self):
        """Start viewer thread, accept one client, and dispatch"""
        if not self.no_viewer:
            t = threading.Thread(target=self.start_viewer, daemon=True)
            t.start()

        sim_t = threading.Thread(target=self.simulation_thread, daemon=True)
        sim_t.start()

        while True:
            client, addr = self.server_socket.accept()
            t = threading.Thread(
                target=self.handle_client,
                args=(client, addr),
                daemon=True
            )
            t.start()

    def close(self):
        """Cleanup sockets and viewer"""
        self.logger.info("Shutting down server...")
        self.running = False
        # Remove client_socket cleanup since it's per-thread now
        if self.server_socket:
            self.server_socket.close()
        if self.viewer:
            self.viewer.close()


def find_model_path():
    """
    Find the MuJoCo model path using a default location
    """
    env_model_path = os.getenv("MUJOCO_MODEL_PATH")
    if env_model_path and os.path.exists(env_model_path):
        return env_model_path
    default_path = "/home/adam/Documents/coding/autonomous/franka_emika_panda/mjx_scene.xml"
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError("Could not find model file at default path")

class MujocoClient:
    def __init__(self):
        self.host = os.getenv("MUJOCO_HOST", "localhost")
        self.port = int(os.getenv("MUJOCO_PORT", 5555))
        self.socket = None
        self.logger = get_logger('MujocoClient')
        
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.logger.info(f"Connected to MuJoCo server at {self.host}:{self.port}")
    
    @staticmethod
    def recv_robot_list():
        """
        Receive the list of robots from the server.
        """
        client = MujocoClient()
        client.connect()
        packet = RobotListPacket(robot_id='robot_list')
        packet = client.send_and_recv(packet)
        if packet is None or not hasattr(packet, 'robot_list'):
            raise ValueError("Failed to receive robot list from server")
        client.close()
        return packet.robot_list
        
    def send_and_recv(self, packet):
        """
        Send a pickled packet and receive the pickled reply, with debug logs.
        """
        if self.socket is None:  # Fixed: Check self.socket instead of sock
            raise ConnectionError("Socket is not connected")
        try:
            self.logger.debug(f"send_and_recv - sending packet: {packet}")
            data_out = pickle.dumps(packet)
            self.socket.sendall(struct.pack('!I', len(data_out)) + data_out)  # Fixed: Use self.socket

            # Read length prefix
            size_data = self.socket.recv(4)  # Fixed: Use self.socket
            if not size_data:
                raise ConnectionError("No data received for length prefix")
            size = struct.unpack('!I', size_data)[0]
            self.logger.debug(f"send_and_recv - expecting {size} bytes reply")
            
            # Read the full reply
            buf = b''
            while len(buf) < size:
                chunk = self.socket.recv(size - len(buf))  # Fixed: Use self.socket
                if not chunk:
                    raise ConnectionError("Incomplete payload: connection closed")
                buf += chunk
            reply = pickle.loads(buf)
            self.logger.debug(f"send_and_recv - received reply: {reply}")
            return reply
        except Exception:
            self.logger.error("Exception in send_and_recv", exc_info=True)
            raise
        
    def close(self):
        if self.socket:
            self.logger.info("Closing MuJoCo client socket")
            self.socket.close()
            self.socket = None  # Added: Reset socket to None after closing
        else:
            self.logger.debug("MuJoCo client socket is already closed or was never opened")  # Changed to debug

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):  # Fixed: Added missing parameters
        self.close()


if __name__ == '__main__':
    model_path = find_model_path()
    server = MuJoCoServer(model_path, no_viewer=True)
    server.run()