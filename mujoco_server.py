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
from packet_example import Packet

import mujoco
import mujoco.viewer

class MuJoCoServer:
    def __init__(self, xml_path, host='localhost', port=5555,
                 render_rgb=True, rgb_width=256, rgb_height=256):
        """
        Initialize MuJoCo server to run physics simulation and communicate
        with a client via TCP sockets.
        Non-server internals are stubbed with dummy values.
        """
        self.host = host
        self.port = port
        self.running = True
        self.locker = threading.Lock()
        self.client_socket = None
        self.server_socket = None

        # Load MuJoCo model (real)
        print(f"Loading MuJoCo model from: {xml_path}")
        model_path = Path(xml_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data  = mujoco.MjData(self.model)
        print("Model and data initialized successfully")

        # Set initial pose (real)
        self.data.qpos[2] = 1.0
        mujoco.mj_forward(self.model, self.data)

        # Viewer settings
        self.render_rgb = render_rgb
        self.rgb_width  = rgb_width
        self.rgb_height = rgb_height
        self.viewer     = None

        # Networking setup
        self.setup_socket()
        self._setup_joint_mappings()



    def setup_socket(self):
        """Set up the TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")



    def _setup_joint_mappings(self):
        """Create dummy joint mapping for stubs"""
        # Stub: one dummy joint
        self.required_joint_names = ['dummy_joint']
        self.required_joint_ids   = [0]



    def start_viewer(self):
        """Launch MuJoCo passive viewer in background thread"""
        try:
            print("Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(0.5)
            print("Viewer initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize viewer: {e}")
            self.viewer = None



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
            return

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

    def _recv_packet(self):
        """Receive a length‑prefixed pickled packet"""
        size_data = self.client_socket.recv(4)
        if not size_data:
            return None
        size = struct.unpack('!I', size_data)[0]
        buf = b''
        while len(buf) < size:
            buf += self.client_socket.recv(size - len(buf))
        return pickle.loads(buf)

    def _send_packet(self, packet):
        """Send a length‑prefixed pickled packet"""
        data = pickle.dumps(packet)
        self.client_socket.sendall(struct.pack('!I', len(data)) + data)

    def handle_client(self, client_socket, addr):
        """
        Handle incoming RPCs: 'action' → apply+step, else → return state
        """
        self.client_socket = client_socket
        print(f"Client connected: {addr}")
        try:
            while True:
                pkt: Packet = self._recv_packet()
                if pkt is None:
                    break
                if pkt.action is not None:
                    self.apply_commands(pkt)
                    self.step_once()
                    reply = {'status': 'ok'}
                else:
                    reply = self.fill_packet(pkt)
                self._send_packet(reply)
        except Exception as e:
            print(f"Error in client loop: {e}")
        finally:
            print("Client disconnected")
            self.close()

    def simulation_thread(self, control_hz=60):
        dt = 1.0 / control_hz
        next_time = time.time()
        while self.running:
            # 1) apply whatever the last action was   #TODO


            
            # 2) step the muJoCo sim
            with self.locker:
                mujoco.mj_step(self.model, self.data)

            # 3) redraw viewer (passive or active)
            self.update_viewer()

            # 4) sleep until next frame
            next_time += dt
            time.sleep(max(0, next_time - time.time()))

    def run(self):
        """Start viewer thread, accept one client, and dispatch"""
        t = threading.Thread(target=self.start_viewer, daemon=True)
        t.start()

        sim_t = threading.Thread(target=self.simulation_thread, daemon=True)
        sim_t.start()

        client, addr = self.server_socket.accept()
        self.handle_client(client, addr)

    def close(self):
        """Cleanup sockets and viewer"""
        print("Shutting down server...")
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        if self.viewer:
            self.viewer.close()
        exit(0)


def find_model_path():
    """
    Find the MuJoCo model path using a default location
    """
    default_path = "/home/adam/Documents/coding/autonomous/franka_emika_panda/mjx_scene.xml"
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError("Could not find model file at default path")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo Server for Robot Control")
    parser.add_argument('--model', type=str, help="Path to MuJoCo XML model file")
    args = parser.parse_args()

    model_path = args.model if args.model else find_model_path()
    server = MuJoCoServer(model_path, host='localhost', port=5555)
    server.run()
