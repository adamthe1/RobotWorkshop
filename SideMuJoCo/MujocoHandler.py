import mujoco
from robot_body_control import RobotBodyControl
from physics_state_extractor import PhysicsStateExtractor
from camera_renderer import CameraRenderer  
from action_manager import ActionManager

import numpy as np
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

class MuJoCoHandler:
    def __init__(self, xml_path, host='localhost', port=5555,
                 render_rgb=True, rgb_width=256, rgb_height=256):
        self.xml_path = xml_path        
        self.host = host
        self.port = port
        self.running = True
        self.locker = threading.Lock()
        self.client_socket = None
        self.server_socket = None
        self.model= None
        self.data = None
        self.robot_control = None


        self.load_scene(xml_path)
        
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
        #self._setup_joint_mappings()




    def setup_socket(self):
        """Set up the TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")




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



    def update_viewer(self):
        """Update the MuJoCo viewer if it's running."""
        if self.viewer is not None and self.viewer.is_running():
            with self.locker:
                self.viewer.sync()



    def load_scene(self,xml_path):
         # Load MuJoCo model (real)
        print(f"Loading MuJoCo model from: {xml_path}")
        model_path = Path(xml_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data  = mujoco.MjData(self.model)
        print("Model and data initialized successfully")
        self.robot_control = RobotBodyControl(self.model, self.data)
        print("RobotBodyControl initialized successfully")





    def step(self):
        with self.locker:
            mujoco.mj_step(self.model, self.data)





    def _recv_packet(self):
        """Receive a length‑prefixed pickled packet"""
        print("DEBUG: _recv_packet in handler - waiting for size data")
        size_data = self.client_socket.recv(4)
        print(f"DEBUG: _recv_packet in handler - received size data: {size_data}")
        if not size_data:
            print("DEBUG: _recv_packet in handler - no size data received, returning None")
            return None
        size = struct.unpack('!I', size_data)[0]
        buf = b''
        while len(buf) < size:
            buf += self.client_socket.recv(size - len(buf))
        return pickle.loads(buf)

    def _send_packet(self, packet):
        """Send a length‑prefixed pickled packet"""
        print(f"DEBUG: _send_packet in handler - sending packet: {packet}")
        data = pickle.dumps(packet)
        self.client_socket.sendall(struct.pack('!I', len(data)) + data)




    def handle_client(self, client_socket, addr):
        """
        Handle incoming RPCs: 'action' → apply+step, else → return state
        """
        print("in handle_client")
        self.client_socket = client_socket
        print(f"Client connected: {addr}")
        try:
            while True:
                pkt: Packet = self._recv_packet()
                if pkt is None:
                    print("No packet received, closing connection.")
                    break
                if pkt.action is not None:
                    print(f"Received action packet: {pkt.action}")
                    self.robot_control.apply_commands(pkt)
                    self.step()
                    reply = {'status': 'ok'}
                else:
                    print("apply robot control")
                    reply = self.robot_control.fill_packet(pkt)
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
        print("in run")
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
    #default_path = "/home/adam/Documents/coding/autonomous/franka_emika_panda/mjx_scene.xml"
    default_path = "/root/RobotWorkshop/franka_emika_panda/mjx_scene.xml"
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError("Could not find model file at default path")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo Server for Robot Control")
    parser.add_argument('--model', type=str, help="Path to MuJoCo XML model file")
    args = parser.parse_args()

    model_path = args.model if args.model else find_model_path()
    server = MuJoCoHandler(model_path, host='localhost', port=5555)
    server.run()