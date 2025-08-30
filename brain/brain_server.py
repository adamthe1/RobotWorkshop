#!/usr/bin/env python3
# filepath: /home/adam/Documents/coding/autonomous/brain/brain_server.py

import os
import sys
import socket
import pickle
import struct
import time
import threading
import signal  # Add signal import
import numpy as np
from pathlib import Path

from mujoco_folder.packet_example import Packet

from logger_config import get_logger
from dotenv import load_dotenv

load_dotenv()

class BrainServer:
    def __init__(self, host=os.getenv("BRAIN_HOST", "localhost"), port=int(os.getenv("BRAIN_PORT", 8900)), use_test_mapper=True):
        """
        Initialize Brain server for policy inference and action generation.
        """
        self.host = host
        self.port = port
        self.running = True
        self.server_socket = None
        self.logger = get_logger('BrainServer')
        
        
        # Optional episode replay mapper or joint test mapper
        self.mapper = None
        
        if use_test_mapper:
            try:
                from .joint_test_mapper import JointTestMapper
                self.mapper = JointTestMapper(
                    num_joints=7,  # Default, will auto-adjust based on robot
                    test_amplitude=1.5,
                    loop=True
                )
                self.logger.info("JointTestMapper loaded for comprehensive joint testing")
            except Exception as e:
                self.logger.error(f"Failed to load JointTestMapper: {e}")
        else:
            replay_path = os.getenv("REPLAY_EPISODE_PATH")
            if replay_path:
                try:
                    from .episode_action_mapper import EpisodeActionMapper
                    self.mapper = EpisodeActionMapper(replay_path)
                    self.logger.info(f"EpisodeActionMapper loaded from {replay_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load EpisodeActionMapper: {e}")

        # Setup networking
        self.setup_socket()
        self.logger.info("Brain server initialized")
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print(f"Logging server received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Clean shutdown of the brain server"""
        self.logger.info("Stopping brain server...")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.warning(f"Error closing brain server socket: {e}")
            finally:
                self.server_socket = None
        self.logger.info("Brain server stopped")
        sys.exit(0)

    def setup_socket(self):
        """Set up the TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        self.logger.info(f"Brain server listening on {self.host}:{self.port}")

    def fill_action(self, packet):
        """
        Generate action based on packet state.
        This is where the actual policy/model inference would happen.
        """
        time_now = time.time()
        robot_id = packet.robot_id
        mission = packet.mission
        
        # If mapper loaded, replay action from episode; else generate dummy
        if self.mapper is not None:
            try:
                action = self.mapper.next_action(
                    robot_id=str(robot_id),
                    joint_names=getattr(packet, 'joint_names', None),
                    qpos=getattr(packet, 'qpos', None),
                    qvel=getattr(packet, 'qvel', None),
                )
            except Exception as e:
                self.logger.error(f"Mapper failed, falling back to dummy: {e}")
                action = None
        else:
            action = None

        if action is None:
            # Fallback: dummy small random action with plausible dim
            if hasattr(packet, 'qpos') and packet.qpos is not None:
                action_dim = len(packet.qpos)
            else:
                action_dim = 7
            action = np.random.uniform(-0.1, 0.1, action_dim).tolist()
        
        
        # Log what we're doing
        self.logger.debug(f"Generating action for robot {robot_id}, mission: {mission}")
        self.logger.debug(f"Action: {action}")
        
        # Fill the action in the packet
        packet.action = action
        time_taken = time.time() - time_now
        self.logger.debug(f"Action generated for robot {robot_id} in {time_taken:.2f} seconds")
        
        return packet

    def _recv_packet(self, client_socket):
        """Receive a length-prefixed pickled packet"""
        size_data = client_socket.recv(4)
        if not size_data:
            return None
        size = struct.unpack('!I', size_data)[0]
        buf = b''
        while len(buf) < size:
            chunk = client_socket.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Connection lost during packet receive")
            buf += chunk
        return pickle.loads(buf)

    def _send_packet(self, packet, client_socket):
        """Send a length-prefixed pickled packet"""
        data = pickle.dumps(packet)
        client_socket.sendall(struct.pack('!I', len(data)) + data)

    def handle_client(self, client_socket, addr):
        """
        Handle incoming requests from clients.
        """
        self.logger.info(f"Brain client connected: {addr}")
        try:
            while self.running:
                pkt = self._recv_packet(client_socket)
                if pkt is None:
                    break
                    
                # Process packet and generate action
                reply = self.fill_action(pkt)
                self._send_packet(reply, client_socket)
                
        except Exception as e:
            self.logger.error(f"Error in brain client loop: {e}")
        finally:
            client_socket.close()
            self.logger.info(f"Brain client {addr} disconnected")

    def run(self):
        """Start accepting clients"""
        self.logger.info("Brain server starting...")
        
        try:
            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                    t = threading.Thread(
                        target=self.handle_client,
                        args=(client, addr),
                        daemon=True
                    )
                    t.start()
                except OSError:
                    if self.running:  # Only log if not intentionally stopping
                        self.logger.error("Brain server socket error")
                    break
        except KeyboardInterrupt:
            self.logger.info("Brain server interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Brain server error: {e}")
        finally:
            self.shutdown()

    def close(self):
        """Cleanup server socket (called by shutdown)"""
        self.shutdown()


class BrainClient:
    def __init__(self):
        self.host = os.getenv("BRAIN_HOST", "localhost")
        self.port = int(os.getenv("BRAIN_PORT", 8900))
        self.socket = None
        self.logger = get_logger('BrainClient')
        
    def connect(self):
        """Connect to brain server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.logger.info(f"Connected to Brain server at {self.host}:{self.port}")
    
    def send_and_recv(self, packet):
        """
        Send a packet to brain server and receive action-filled packet back.
        """
        if self.socket is None:
            raise ConnectionError("Socket is not connected")
        try:
            self.logger.debug(f"Brain send_and_recv - sending packet: {packet}")
            data_out = pickle.dumps(packet)
            self.socket.sendall(struct.pack('!I', len(data_out)) + data_out)

            # Read length prefix
            size_data = self.socket.recv(4)
            if not size_data:
                raise ConnectionError("No data received for length prefix")
            size = struct.unpack('!I', size_data)[0]
            self.logger.debug(f"Brain send_and_recv - expecting {size} bytes reply")
            
            # Read the full reply
            buf = b''
            while len(buf) < size:
                chunk = self.socket.recv(size - len(buf))
                if not chunk:
                    raise ConnectionError("Incomplete payload: connection closed")
                buf += chunk
            reply = pickle.loads(buf)
            self.logger.debug(f"Brain send_and_recv - received reply with action: {reply.action}")
            return reply
        except Exception:
            self.logger.error("Exception in brain send_and_recv", exc_info=True)
            raise
        
    def close(self):
        """Close connection to brain server"""
        if self.socket:
            self.logger.info("Closing Brain client socket")
            self.socket.close()
            self.socket = None
        else:
            self.logger.debug("Brain client socket is already closed or was never opened")

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    server = BrainServer()
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down brain server...")
        server.close()
