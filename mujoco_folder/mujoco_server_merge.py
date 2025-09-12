#!/usr/bin/env python3
# MuJoCo Server Script (dummy state stubs)
# This script initializes MuJoCo, runs physics simulation, and communicates
# with a client via TCP sockets. Non-server functionality is stubbed.


from .robot_body_control import RobotBodyControl
from .physics_state_extractor import PhysicsStateExtractor
from .camera_renderer import CameraRenderer  
from .action_manager import ActionManager
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
from .compose_scene import save_xml_file
from logger_config import get_logger
import atexit
import signal

from dotenv import load_dotenv
import os
import random

load_dotenv()

class MuJoCoServer:
    def __init__(self, xml_path=None, host=os.getenv("MUJOCO_HOST", "localhost"), port=int(os.getenv("MUJOCO_PORT", 5555)),
                 render_rgb=True, rgb_width=256, rgb_height=256, num_robots=3):
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

        if os.getenv('USE_GPU_WSL', '0') == '1':
            os.environ['MUJOCO_GL'] = 'egl'
            self.logger.info("Using GPU rendering (EGL)")
        else:
            self.logger.info("Using CPU rendering (OSMesa)")

        self.robot_ids = ['r1']
        self.robot_dict = {"r1": "FrankaPanda"}
        self.load_scene(xml_path, num_robots=num_robots)
        
        # Optionally load saved state from disk
        loaded = self.load_saved_state_if_enabled()
        if not loaded:
            # Fallback initial tweak if no saved state was loaded
            self.data.qpos[2] = 1.0
            mujoco.mj_forward(self.model, self.data)
        
        # Initialize action cache with current control values to prevent robots from falling
        # This must happen after any state loading to capture the correct initial positions
        self.robot_control.initialize_action_cache_from_current_state()
        self.logger.info("Action cache initialized with current robot positions")
        
        # Viewer settings
        self.render_rgb = render_rgb
        self.rgb_width  = rgb_width
        self.rgb_height = rgb_height
        self.viewer     = None

        self.control_hz = 60  # Control frequency in Hz

        self.no_viewer = int(os.getenv("NO_VIEWER", 0)) 
        self.logger.info(f"MuJoCo server initialized on {self.host}:{self.port} with no_viewer={self.no_viewer}")
        

        # Networking setup
        self.setup_socket()
        atexit.register(self.cleanup_on_exit)
        # Note: main process handles SIGINT/SIGTERM; server exits via terminal


    def load_scene(self, xml_path, num_robots=None):  # Remove num_robots parameter since we use env vars
        if os.getenv("USE_DUPLICATING_PATH", "0") == "1":
            save_xml = os.getenv("MAIN_DIRECTORY") + "/xml_robots/panda_scene.xml"
            xml_path, robot_dict = save_xml_file(save_xml)  # Now returns robot_dict
            self.robot_dict = robot_dict  # Store robot type mapping
            self.robot_ids = list(robot_dict.keys())  # Extract robot IDs
            
        # Load MuJoCo model
        self.logger.info(f"Loading MuJoCo model from: {xml_path}")
        model_path = Path(xml_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.logger.info("Model and data initialized successfully")

        # Pass robot dict to RobotBodyControl
        self.robot_control = RobotBodyControl(self.model, self.data, robot_dict=self.robot_dict)
        self.logger.info(f"RobotBodyControl initialized with robots: {self.robot_dict}")




    def load_saved_state_if_enabled(self) -> bool:
        """Load a single-robot joint-only state and apply to all robots.

        When LOAD_SAVED_STATE is enabled, finds the newest .npz under
        xml_robots/saved_state containing keys:
          - joint_names (base names, e.g., 'joint1', 'finger_joint1', ...)
          - qpos, qvel (same length as joint_names)

        For each robot listed in self.robot_ids, prefixes each base joint name
        (e.g., 'r1_joint1') and applies hinge/slide joint position/velocity.
        Returns True if anything was applied, else False.
        """
        flag = os.getenv("LOAD_SAVED_STATE", "0").strip().lower()
        if flag not in ("1", "true", "yes", "on"):  # disabled
            return False
        try:
            base = os.getenv("MAIN_DIRECTORY") or str(Path.cwd())
            # Allow explicit override via env
            override = os.getenv("REPLAY_SAVED_STATE_DIR", "").strip()
            search_dirs = []
            if override:
                search_dirs.append(Path(override))
            search_dirs.append(Path(base) / "xml_robots" / "saved_state")
            search_dirs.append(Path(base) / "finetuning" / "saved_robot_states")

            # Collect all candidate .npz files across directories
            candidates = []
            for d in search_dirs:
                if d.exists():
                    candidates.extend(list(d.glob("*.npz")))

            if not candidates:
                self.logger.warning(
                    "LOAD_SAVED_STATE set but no .npz files found in: " + ", ".join(str(p) for p in search_dirs)
                )
                return False
            # Use most recent by mtime
            candidates.sort(key=lambda p: p.stat().st_mtime)
            path = candidates[-1]
            arr = np.load(path, allow_pickle=True)
            if not ("joint_names" in arr and "qpos" in arr and "qvel" in arr):
                self.logger.warning(f"Saved file {path} does not contain joint_names/qpos/qvel; skipping")
                return False
            names = [str(x) for x in arr["joint_names"].tolist()]
            qpos = np.array(arr["qpos"], dtype=float).reshape(-1)
            qvel = np.array(arr["qvel"], dtype=float).reshape(-1)
            if not (len(names) == len(qpos) == len(qvel)):
                self.logger.warning(f"Saved joint arrays length mismatch in {path}")
                return False

            applied = 0
            for rid in self.robot_ids:
                prefix = f"{rid}_"
                for nm, qp, qv in zip(names, qpos, qvel):
                    full = prefix + nm
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full)
                    if jid < 0:
                        continue
                    jtype = int(self.model.jnt_type[jid])
                    if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                        continue
                    qposadr = int(self.model.jnt_qposadr[jid])
                    dofadr = int(self.model.jnt_dofadr[jid])
                    self.data.qpos[qposadr] = float(qp)
                    self.data.qvel[dofadr] = float(qv)
                    applied += 1

            if applied == 0:
                self.logger.warning(f"No matching joints found when applying {path} to robots {self.robot_ids}")
                return False
            mujoco.mj_forward(self.model, self.data)
            self.logger.info(f"Applied joint-only saved state from {path} to {len(self.robot_ids)} robots; joints set: {applied}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load saved state: {e}")
            return False

    def setup_socket(self):
        """Set up the TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        


    def start_viewer(self):
        """Launch lightweight GLFW viewer and render in this thread.

        Important: OpenGL contexts are thread-affine. We both create and render
        from the same thread here to avoid a black screen.
        """
        try:
            from .lightweight_viewer import LightweightViewer

            if self.viewer is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None
                time.sleep(0.1)  # Brief pause for cleanup

            self.logger.info("Launching lightweight MuJoCo GLFW viewer...")
            self.viewer = LightweightViewer(self.model, self.data, 800, 600, "MuJoCo Viewer").launch()
            self.logger.info("Lightweight viewer initialized successfully; entering render loop")

            # Render loop at ~60 FPS in this thread
            dt = 1.0 / self.control_hz
            while self.running and self.viewer.is_running():
                with self.locker:
                    # Render the current scene safely while physics may update
                    self.viewer.sync()
                time.sleep(dt)

            # Viewer loop ended; leave server running until terminal exit

        except Exception as e:
            self.logger.error(f"Failed to initialize lightweight viewer: {e}")
            self.viewer = None
    
    def cleanup_on_exit(self):
        pass
        

    def fill_robot_list(self, packet):
        """
        Fill the packet with a dummy robot list.
        This is a stub method that simulates receiving a list of robots.
        """
        packet.robot_list = self.get_robot_list()
        packet.robot_dict = self.get_robot_dict()
        return packet
        
    def get_robot_list(self):
        """Return a list of robot IDs (stubbed)"""
        return self.robot_ids

    def get_robot_dict(self):
        """Return a dictionary mapping robot IDs to their types (stubbed)"""
        return self.robot_dict

    def step_once(self):
        """Advance the physics by one timestep"""
        with self.locker:
            mujoco.mj_step(self.model, self.data)

    def update_viewer(self):
        """Update the MuJoCo viewer if it's running."""
        if self.viewer is not None and self.viewer.is_running():
            with self.locker:
                self.viewer.sync()

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
                    self.robot_control.apply_commands(pkt)
                    reply = pkt
                else:
                    # Allow disabling camera images in state to reduce payload
                    no_cam = os.getenv("NO_CAMERA_IN_STATE", "0").strip() in ("1", "true", "yes", "on")
                    reply = self.robot_control.fill_packet(pkt, no_camera=no_cam)
                self._send_packet(reply, client_socket)  # Pass socket
        except Exception as e:
            self.logger.error(f"Error in client loop: {e}")
        finally:
            client_socket.close()  # Close this specific client
            self.logger.info(f"Client {addr} disconnected")

    def simulation_thread(self):
        dt = 1.0 / self.control_hz
        next_time = time.time()
        
        # Enable action repeat mode for gravity compensation
        self.robot_control.set_action_repeat_mode(True)
        
        while self.running:
            with self.locker:
                # 1) Commit staged actions to data before stepping
                actions_applied = self.robot_control.commit_staged_actions()
                
                # 2) Step the MuJoCo simulation
                mujoco.mj_step(self.model, self.data)
                
                # 3) Update snapshot after stepping (critical: while still holding lock)
                self.robot_control.update_snapshot()

            # 4) Viewer renders from its own thread that owns the GL context
            # (no changes needed here)

            # 5) Sleep until next frame
            next_time += dt
            time.sleep(max(0, next_time - time.time()))

    def run(self):
        """Start viewer thread, clients, and dispatch"""
        try:
            self.logger.info(f"With viewer: {not self.no_viewer}")
            if self.no_viewer == 0:
                t = threading.Thread(target=self.start_viewer, daemon=True)
                t.start()

            sim_t = threading.Thread(target=self.simulation_thread, daemon=True)
            sim_t.start()

            self.logger.info(f"Server listening on {self.host}:{self.port}")
            while True:
                client, addr = self.server_socket.accept()
                t = threading.Thread(
                    target=self.handle_client,
                    args=(client, addr),
                    daemon=True
                )
                t.start()
                
        except KeyboardInterrupt:
            self.logger.info("Server interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.close()

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
    
    @staticmethod
    def recv_robot_list_and_dict():
        """
        Receive the list of robots and their types from the server.
        """
        client = MujocoClient()
        client.connect()
        packet = RobotListPacket(robot_id='robot_list')
        packet = client.send_and_recv(packet)
        if packet is None or not hasattr(packet, 'robot_dict'):
            raise ValueError("Failed to receive robot dict from server")
        client.close()
        return packet.robot_list, packet.robot_dict

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
    
    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    server = MuJoCoServer(xml_path=find_model_path(), num_robots=1)
    server.run()
