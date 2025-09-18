#!/usr/bin/env python3
# MuJoCo Server Script (dummy state stubs)
# This script initializes MuJoCo, runs physics simulation, and communicates
# with a client via TCP sockets. Non-server functionality is stubbed.


from mujoco_folder.robot_control.robot_body_control import RobotBodyControl
from mujoco_folder.robot_control.physics_state_extractor import PhysicsStateExtractor
from mujoco_folder.robot_control.camera_renderer import CameraRenderer  
from mujoco_folder.robot_control.action_manager import ActionManager
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
from mujoco_folder.scene_control.compose_scene import save_xml_file
from mujoco_folder.scene_control.reset_cup import (
    reset_cup_and_bottles_to_default,
)
from logger_config import get_logger
import atexit
import signal

from dotenv import load_dotenv
import os
import random

load_dotenv()

class MuJoCoServer:
    def __init__(self, host=os.getenv("MUJOCO_HOST", "localhost"), port=int(os.getenv("MUJOCO_PORT", 5555)),
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
        self.load_scene()
        
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

        # Control frequency in Hz (shared across system via CONTROL_HZ)
        try:
            self.control_hz = float(os.getenv("CONTROL_HZ", "30"))
        except Exception:
            self.control_hz = 30.0

        self.no_viewer = int(os.getenv("NO_VIEWER", 0)) 
        self.logger.info(f"MuJoCo server initialized on {self.host}:{self.port} with no_viewer={self.no_viewer}")
        

        # Networking setup
        self.setup_socket()
        atexit.register(self.cleanup_on_exit)
        # Note: main process handles SIGINT/SIGTERM; server exits via terminal


    def load_scene(self):
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
        """
        Load a joint-only state for each robot type (FrankaPanda, SO101) and apply to all robots of that type.
        Looks for saved_panda.npz and saved_so101.npz in xml_robots/saved_state.
        """
        flag = os.getenv("LOAD_SAVED_STATE", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return False

        try:
            base = os.getenv("MAIN_DIRECTORY") or str(Path.cwd())
            state_dir = Path(base) / "xml_robots" / "saved_state"
            type_to_file = {
                "FrankaPanda": state_dir / "saved_panda.npz",
                "SO101": state_dir / "saved_so101.npz",
            }

            applied = 0
            for rid, rtype in self.robot_dict.items():
                npz_path = type_to_file.get(rtype)
                if not npz_path or not npz_path.exists():
                    self.logger.warning(f"No saved state file for robot {rid} of type {rtype} at {npz_path}")
                    continue

                arr = np.load(npz_path, allow_pickle=True)
                if not ("joint_names" in arr and "qpos" in arr and "qvel" in arr):
                    self.logger.warning(f"Saved file {npz_path} missing joint_names/qpos/qvel; skipping")
                    continue

                names = [str(x) for x in arr["joint_names"].tolist()]
                self.logger.info(f"Applying saved state to robot {rid} of type {rtype} with joints: {names}")
                qpos = np.array(arr["qpos"], dtype=float).reshape(-1)
                qvel = np.array(arr["qvel"], dtype=float).reshape(-1)
                if not (len(names) == len(qpos) == len(qvel)):
                    self.logger.warning(f"Saved joint arrays length mismatch in {npz_path}")
                    continue

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
                self.logger.warning("No matching joints found when applying saved states to robots")
                return False
            mujoco.mj_forward(self.model, self.data)
            self.logger.info(f"Applied saved states to robots; joints set: {applied}")
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
            from mujoco_folder.scene_control.lightweight_viewer import LightweightViewer

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
                elif getattr(pkt, 'mission', None) == 'reset_the_scene_cup_bottles':
                    try:
                        # Packet carries robot_id as a string; extract index via split('_')[-1]
                        reset_cup_and_bottles_to_default(self.model, self.data, pkt.robot_id)
                        reply = pkt
                    except Exception as e:
                        self.logger.error(f"reset_cup failed for robot_id={pkt.robot_id}: {e}")
                        reply = pkt
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
        """Run the server: viewer in main thread, simulation and networking in background threads."""
        try:
            self.logger.info(f"With viewer: {not self.no_viewer}")
            self.start_viewer()

            # Start simulation in a background thread
            sim_t = threading.Thread(target=self.simulation_thread, daemon=True)
            sim_t.start()

            # Start networking in a background thread
            net_t = threading.Thread(target=self.network_thread, daemon=True)
            net_t.start()

            # Main thread: run the viewer loop (blocking, processes events and calls sync)
            if self.viewer is not None and self.viewer.is_running():
                self.logger.info("Entering viewer loop in main thread")
                while self.viewer.is_running():
                    with self.locker:
                        self.viewer.sync()
                    if self.viewer.exit_requested():
                        self.logger.info("Viewer requested exit, shutting down server.")
                        raise KeyboardInterrupt
                    time.sleep(1.0 / self.control_hz)  # Or whatever is smooth for your viewer

        except KeyboardInterrupt:
            print("Server interrupted by user")
            self.logger.info("Server interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.close()

    def network_thread(self):
        """Accept clients and spawn handler threads (runs in background)."""
        self.server_socket.settimeout(0.1)
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                t = threading.Thread(
                    target=self.handle_client,
                    args=(client, addr),
                    daemon=True
                )
                t.start()
            except socket.timeout:
                continue

    def close(self):
        """Cleanup sockets and viewer"""
        self.logger.info("Shutting down server...")
        self.running = False
        # Remove client_socket cleanup since it's per-thread now
        if self.server_socket:
            self.server_socket.close()
        if self.viewer:
            self.viewer.close()


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

    # Convenience RPCs
    @staticmethod
    def reset_cup(robot_id: str):
        """Request the server to reset the beer glass for the given robot_id.

        robot_id example: 'panda_0' or 'so101_0'. The server will parse the trailing
        number and map to beer_glass_free{index+1}.
        """
        client = MujocoClient()
        client.connect()
        pkt = Packet(robot_id=robot_id, mission='reset_the_scene_cup_bottles')
        try:
            _ = client.send_and_recv(pkt)
        finally:
            client.close()


if __name__ == '__main__':
    server = MuJoCoServer()
    server.run()
