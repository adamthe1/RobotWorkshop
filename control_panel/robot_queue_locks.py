import socket
import threading
import json
import queue
from collections import deque
import struct
from dotenv import load_dotenv
import os
from logger_config import get_logger
import time
load_dotenv()

def _recvall(sock, n):
    """Helper to receive exactly n bytes"""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

class QueueServer:
    def __init__(self):
        self.host = os.getenv("QUEUE_HOST", "localhost")
        self.port = int(os.getenv("QUEUE_PORT", 9000))
        
        # Initialize data structures
        self.robot_locks = {}  # 0 = free, 1 = locked
        self.mission_queue = deque()
        self.free_robot_queue = deque()  # Initially all robots are free
        
        # Thread locks for thread safety
        self.locks_lock = threading.Lock()
        self.mission_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        
        self.running = False
        self.server_socket = None
        self.logger = get_logger('QueueServer')    # ← add this
        
        
    def start_server(self):
        """Start the queue server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        try:
            self.logger.info(f"Queue server listening on {self.host}:{self.port}")
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                except OSError:
                    if self.running:  # Only log if not intentionally stopping
                        self.logger.error("Server socket error")
                    break
        except KeyboardInterrupt:
            self.logger.info("Queue server interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Queue server error: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the server."""
        self.logger.info("Stopping queue server...")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.warning(f"Error closing queue server socket: {e}")
            finally:
                self.server_socket = None
        self.logger.info("Queue server stopped")
                
    def handle_client(self, client_socket, address):
        """Handle client requests with proper message framing."""
        try:
            while True:
                # Read 4-byte length prefix
                raw_len = _recvall(client_socket, 4)
                if not raw_len:
                    break
                msg_len = struct.unpack('!I', raw_len)[0]
                
                # Read the JSON payload
                data = _recvall(client_socket, msg_len)
                if not data:
                    break
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    response = self.process_request(request)
                    
                    # Send response with length prefix
                    resp_bytes = json.dumps(response).encode('utf-8')
                    client_socket.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)
                    
                except json.JSONDecodeError:
                    error_response = {"status": "error", "message": "Invalid JSON"}
                    resp_bytes = json.dumps(error_response).encode('utf-8')
                    client_socket.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)
                    
        except (ConnectionResetError, BrokenPipeError):
            self.logger.error(f"Client {address} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            
    def process_request(self, request):
        """Process client requests and return responses."""
        action = request.get("action")
        
        if action == "enqueue_mission":
            mission = request.get("mission")
            return self.enqueue_mission(mission)
            
        elif action == "dequeue_mission":
            return self.dequeue_mission()
            
        elif action == "enqueue_robot":
            robot_id = request.get("robot_id")
            return self.enqueue_robot(robot_id)
            
        elif action == "dequeue_robot":
            return self.dequeue_robot()
        
        elif action == "see_next_robot":
            return self.see_next_robot()
        
        elif action == "get_robot_mission_pair": 
            return self.get_robot_mission_pair()

        elif action == "get_robot_lock":
            robot_id = request.get("robot_id")
            return self.get_robot_lock(robot_id)
            
        elif action == "set_robot_lock":
            robot_id = request.get("robot_id")
            lock_state = request.get("lock_state")
            return self.set_robot_lock(robot_id, lock_state)
        
        elif action == "add_robot_ids":  # New action
            robot_ids = request.get("robot_ids")
            return self.add_robot_ids(robot_ids)
            
        elif action == "get_status":
            return self.get_status()
            
        else:
            return {"status": "error", "message": "Unknown action"}
        
    def add_robot_ids(self, robot_ids):
        """Add multiple robot IDs to the server with locks and free queue."""
        if not isinstance(robot_ids, list):
            return {"status": "error", "message": "robot_ids must be a list"}
        
        added_robots = []
        skipped_robots = []
        
        with self.locks_lock, self.robot_lock:
            for robot_id in robot_ids:
                if robot_id not in self.robot_locks:
                    # Add to locks (free state)
                    self.robot_locks[robot_id] = 0
                    # Add to free robot queue
                    self.free_robot_queue.append(robot_id)
                    added_robots.append(robot_id)
                else:
                    skipped_robots.append(robot_id)
        self.logger.info(f"Added robots: {added_robots}, Skipped robots: {skipped_robots}")
        return {
            "status": "success",
            "message": f"Added {len(added_robots)} robots",
            "added_robots": added_robots,
            "skipped_robots": skipped_robots
        }
    
    def get_robot_mission_pair(self):
        """Get a robot-mission pair atomically with proper locking."""
        with self.robot_lock, self.mission_lock, self.locks_lock:
            # Check if both robot and mission are available
            if not self.free_robot_queue:
                return {"status": "no_robots", "robot_id": None, "mission": None}
            
            if not self.mission_queue:
                return {"status": "no_missions", "robot_id": None, "mission": None}
            
            # Get robot and mission atomically
            robot_id = self.free_robot_queue.popleft()
            mission = self.mission_queue.popleft()
            
            # Set robot lock to locked
            self.robot_locks[robot_id] = 1
            
            return {
                "status": "success", 
                "robot_id": robot_id, 
                "mission": mission
            }
        
    def enqueue_mission(self, mission):
        """Add mission to queue."""
        with self.mission_lock:
            self.mission_queue.append(mission)
            return {"status": "success", "message": f"Mission '{mission}' enqueued"}
    
    def dequeue_mission(self):
        """Get next mission from queue."""
        with self.mission_lock:
            if self.mission_queue:
                mission = self.mission_queue.popleft()
                return {"status": "success", "mission": mission}
            else:
                return {"status": "empty", "mission": None}
    
    def enqueue_robot(self, robot_id):
        """Add robot to free robot queue."""
        with self.robot_lock:
            if robot_id not in self.free_robot_queue:
                self.free_robot_queue.append(robot_id)
                # Also set robot lock to free
                with self.locks_lock:
                    self.robot_locks[robot_id] = 0
                return {"status": "success", "message": f"Robot '{robot_id}' enqueued"}
            else:
                return {"status": "error", "message": f"Robot '{robot_id}' already in queue"}
    
    def dequeue_robot(self):
        """Get next free robot from queue."""
        with self.robot_lock:
            if self.free_robot_queue:
                robot_id = self.free_robot_queue.popleft()
                # Set robot lock to locked
                with self.locks_lock:
                    self.robot_locks[robot_id] = 1
                return {"status": "success", "robot_id": robot_id}
            else:
                return {"status": "empty", "robot_id": None}
            
    def see_next_robot(self):
        """See next robot from queue without removing it."""
        with self.robot_lock:
            if self.free_robot_queue:
                robot_id = self.free_robot_queue[0]  # Peek at first element
                return {"status": "success", "robot_id": robot_id}
            else:
                return {"status": "empty", "robot_id": None}

    
    def get_robot_lock(self, robot_id):
        """Get robot lock state."""
        with self.locks_lock:
            if robot_id in self.robot_locks:
                return {"status": "success", "robot_id": robot_id, "lock_state": self.robot_locks[robot_id]}
            else:
                return {"status": "error", "message": f"Robot '{robot_id}' not found"}
    
    def set_robot_lock(self, robot_id, lock_state):
        """Set robot lock state."""
        with self.locks_lock:
            if robot_id in self.robot_locks:
                self.robot_locks[robot_id] = lock_state
                return {"status": "success", "message": f"Robot '{robot_id}' lock set to {lock_state}"}
            else:
                return {"status": "error", "message": f"Robot '{robot_id}' not found"}
    
    def get_status(self):
        """Get current server status."""
        with self.mission_lock, self.robot_lock, self.locks_lock:
            return {
                "status": "success",
                "mission_queue_length": len(self.mission_queue),
                "free_robot_queue_length": len(self.free_robot_queue),
                "robot_locks": dict(self.robot_locks),
                "free_robots": list(self.free_robot_queue)
            }


class QueueClient:
    def __init__(self):
        self.host = os.getenv("QUEUE_HOST", "localhost")
        self.port = int(os.getenv("QUEUE_PORT", 9000))
        self.logger = get_logger('QueueClient')
        self.socket = None
        
    def connect(self):
        """Connect to the queue server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            return True
        except ConnectionRefusedError:
            self.logger.error(f"Could not connect to queue server at {self.host}:{self.port}")
            time.sleep(1)  # Retry after a short delay
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def _send_request(self, request, tries=0):
        """Send request to server and get response with proper framing."""
        if not self.socket and not self.connect():
            return {"status": "error", "message": "Not connected to server"}
        
        try:
            self.logger.debug(f"Sending request: {request}")
            # Send with length prefix
            payload = json.dumps(request).encode('utf-8')
            self.socket.sendall(struct.pack('!I', len(payload)) + payload)
            
            # Read response length
            raw_len = _recvall(self.socket, 4)
            if not raw_len:
                self.logger.error("No response length received")
                return {"status": "error", "message": "No response length"}
            resp_len = struct.unpack('!I', raw_len)[0]
            
            # Read response payload
            resp_data = _recvall(self.socket, resp_len)
            if not resp_data:
                self.logger.error("No response data received")
                return {"status": "error", "message": "Incomplete response"}
            
            self.logger.debug(f"Received response: {resp_data.decode('utf-8')}")
            return json.loads(resp_data.decode('utf-8'))
            
        except (ConnectionResetError, BrokenPipeError) as e:
            self.logger.warning(f"Connection error: {e}, attempting reconnect")
            # Try to reconnect once
            self.socket = None
            if self.connect():
                self.logger.info("Reconnected to server, retrying request")
                return self._send_request(request)  # Recursive retry
            else:
                self.logger.error("Failed to reconnect to server")
                return {"status": "error", "message": "Connection lost"}
        except Exception as e:
            self.logger.error(f"Request error: {e} {request}")
            if tries < 3:
                self.logger.info(f"Retrying request ({tries + 1}/3)")
                time.sleep(1)
                return self._send_request(request, tries + 1)
            else:
                return {"status": "error", "message": f"Request failed: {e}"}

    
    def get_robot_mission_pair(self):
        """Get a robot-mission pair atomically."""
        request = {"action": "get_robot_mission_pair"}
        response = self._send_request(request)
        
        if response.get("status") == "success":
            self.logger.info(f"Got robot-mission pair: {response}")
            return {
                "robot_id": response.get("robot_id"),
                "mission": response.get("mission")
            }
        elif response.get("status") == "no_robots":
            return {"robot_id": None, "mission": None, "error": "No free robots available"}
        elif response.get("status") == "no_missions":
            return {"robot_id": None, "mission": None, "error": "No missions available"}
        else:
            return {"robot_id": None, "mission": None, "error": "Unknown error"}
    
    def enqueue_mission(self, mission):
        """Add mission to queue."""
        request = {"action": "enqueue_mission", "mission": mission}
        self.logger.info(f"Enqueuing mission: {mission}")
        return self._send_request(request)
    
    def get_mission_from_queue(self):
        """Get next mission from queue."""
        request = {"action": "dequeue_mission"}
        response = self._send_request(request)
        if response.get("status") == "success":
            return response.get("mission")
        return None
    
    def get_robot_from_queue(self):
        """Get next free robot from queue."""
        request = {"action": "dequeue_robot"}
        response = self._send_request(request)
        if response.get("status") == "success":
            return response.get("robot_id")
        return None
        
    
    def see_next_robot(self):
        """See next robot from queue without removing it."""
        request = {"action": "see_next_robot"}
        response = self._send_request(request)
        if response.get("status") == "success":
            return response.get("robot_id")
        return None
    
    def get_robot_lock(self, robot_id):
        """Get robot lock state (0=free, 1=locked)."""
        request = {"action": "get_robot_lock", "robot_id": robot_id}
        response = self._send_request(request)
        if response.get("status") == "success":
            return response.get("lock_state")
        return None
    
    def set_robot_lock(self, robot_id, lock_state):
        """Set robot lock state (0=free, 1=locked)."""
        request = {"action": "set_robot_lock", "robot_id": robot_id, "lock_state": lock_state}
        return self._send_request(request)
    
    def enqueue_robot(self, robot_id):
        """Release robot back to free robot queue."""
        request = {"action": "enqueue_robot", "robot_id": robot_id}
        return self._send_request(request)
    
    def add_robot_ids(self, robot_ids):
        """Add multiple robot IDs to the server."""
        request = {"action": "add_robot_ids", "robot_ids": robot_ids}
        return self._send_request(request)
    
    def get_status(self):
        """Get server status."""
        request = {"action": "get_status"}
        return self._send_request(request)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


if __name__ == '__main__':
    server = QueueServer()
    server.start_server()