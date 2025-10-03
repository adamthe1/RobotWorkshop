import socket
import threading
import json
import queue
from collections import deque, defaultdict
import struct
from dotenv import load_dotenv
import os
from logger_config import get_logger
import time
from control_panel.missions import SUPPORTED_MISSIONS_PER_ROBOT
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
        self.robot_locks = {}  # robot_id -> 0 = free, 1 = locked
        self.robot_dict = {}   # robot_id -> robot_type
        self.mission_queue = deque()  # missions waiting
        self.free_robot_queue = deque()  # generic queue before types known (back-compat)
        # Per-type available robots queues
        self.type_to_queue = defaultdict(deque)  # robot_type -> deque(robot_id)
        # Stats for heuristic
        self.type_missions_given = defaultdict(int)  # robot_type -> count
        # Cached next robot-mission pair to hand out
        self.next_pair = None  # tuple(robot_id, mission) or None
        # Heuristic weights
        self.w1 = 1.0
        self.w2 = 1.0
        
        # Thread locks for thread safety
        self.locks_lock = threading.Lock()
        self.mission_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        self.pair_lock = threading.Lock()
        
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
        # Start background filler to maintain next_pair
        filler_thread = threading.Thread(target=self._filler_loop, daemon=True)
        filler_thread.start()
        
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

        elif action == "set_robot_dict":
            robot_dict = request.get("robot_dict", {})
            return self.set_robot_dict(robot_dict)

        elif action == "get_robot_lock":
            robot_id = request.get("robot_id")
            return self.get_robot_lock(robot_id)
            
        elif action == "set_robot_lock":
            robot_id = request.get("robot_id")
            lock_state = request.get("lock_state")
            return self.set_robot_lock(robot_id, lock_state)
        
        elif action == "add_robot_ids": 
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

    def set_robot_dict(self, robot_dict):
        """Set robot_id -> robot_type mapping and initialize per-type queues.

        Any robots already known and free are moved into their type-specific queues.
        """
        if not isinstance(robot_dict, dict):
            return {"status": "error", "message": "robot_dict must be a dict"}

        with self.robot_lock, self.locks_lock:
            self.robot_dict.update(robot_dict)
            # Ensure locks exist for all robots in dict
            for rid in robot_dict.keys():
                if rid not in self.robot_locks:
                    self.robot_locks[rid] = 0
            # Move any generic free robots into their type queues
            remaining_generic = deque()
            # move all free robots to type queues if type known else keep generic
            while self.free_robot_queue:
                rid = self.free_robot_queue.popleft()
                rtype = self.robot_dict.get(rid)
                if rtype is not None:
                    if self.robot_locks.get(rid, 0) == 0:
                        self.type_to_queue[rtype].append(rid)
                else:
                    remaining_generic.append(rid)

            self.free_robot_queue = remaining_generic
        self.logger.info(f"Robot dict set. Types: {set(robot_dict.values())}")
        return {"status": "success"}
    
    def get_robot_mission_pair(self):
        """Get a robot-mission pair atomically with proper locking."""
        with self.pair_lock, self.robot_lock, self.mission_lock, self.locks_lock:
            if self.next_pair is None:
                # Reflect specific shortage states
                if not self.mission_queue:
                    return {"status": "no_missions", "robot_id": None, "mission": None}
                # Check if any type queue has robots
                any_robot = any(len(q) > 0 for q in self.type_to_queue.values()) or len(self.free_robot_queue) > 0
                if not any_robot:
                    return {"status": "no_robots", "robot_id": None, "mission": None}
                return {"status": "pending", "robot_id": None, "mission": None}

            # Have a pair ready
            robot_id, mission = self.next_pair
            # Lock robot and clear next_pair
            self.robot_locks[robot_id] = 1
            self.next_pair = None
            return {"status": "success", "robot_id": robot_id, "mission": mission}
        
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
        with self.robot_lock, self.locks_lock:
            # Set robot lock to free
            self.robot_locks[robot_id] = 0
            rtype = self.robot_dict.get(robot_id)
            # if i know the type, put in type queue
            if rtype is not None: 
                # Deduplicate within type queue
                if robot_id not in self.type_to_queue[rtype]:
                    self.type_to_queue[rtype].append(robot_id)
                return {"status": "success", "message": f"Robot '{robot_id}' enqueued (type {rtype})"}
            else: # dont know type, put in generic queue
                # Deduplicate within generic queue
                if robot_id not in self.free_robot_queue:
                    self.free_robot_queue.append(robot_id)
                    return {"status": "success", "message": f"Robot '{robot_id}' enqueued (generic)"}
                return {"status": "error", "message": f"Robot '{robot_id}' already in queue"}
    
    def dequeue_robot(self):
        """Get next free robot from queue."""
        with self.robot_lock, self.locks_lock:
            # Prefer type queues if available
            for q in self.type_to_queue.values():
                if q:
                    robot_id = q.popleft()
                    self.robot_locks[robot_id] = 1
                    return {"status": "success", "robot_id": robot_id}
            if self.free_robot_queue:
                robot_id = self.free_robot_queue.popleft()
                self.robot_locks[robot_id] = 1
                return {"status": "success", "robot_id": robot_id}
            return {"status": "empty", "robot_id": None}
            
    def see_next_robot(self):
        """See next robot from queue without removing it."""
        with self.pair_lock:
            if self.next_pair is not None:
                robot_id, _ = self.next_pair
                return {"status": "success", "robot_id": robot_id}
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

    def _eligible_types_for_mission(self, mission):
        """ Build inverse map mission -> types """
        types = []
        for rtype, missions in SUPPORTED_MISSIONS_PER_ROBOT.items():
            if mission in missions:
                types.append(rtype)
        return types

    def _num_supported_missions_for_type(self, rtype):
        missions = SUPPORTED_MISSIONS_PER_ROBOT.get(rtype, [])
        return len(missions)

    def _filler_loop(self):
        """Continuously try to populate next_pair from queues using heuristic.
        Heuristic:
            - Prefer robots that can do fewer missions (more specialized)
            - Prefer robots that have done fewer missions so far

        Scans the mission queue to find the first mission that has at least one
        eligible robot available. This avoids head-of-line blocking.
        """
        while self.running:
            with self.pair_lock:
                if self.next_pair is not None:
                    # Already have a pair ready
                    pass
                else:
                    # Try to form a pair
                    with self.mission_lock, self.robot_lock:
                        if not self.mission_queue:
                            self.logger.debug("no missions in queue yet")
                            time.sleep(0.03)
                            continue

                        # Scan the queue for the first mission with any eligible robot
                        selected = None  # tuple(index, best_type, robot_id, mission)
                        for idx, mission in enumerate(self.mission_queue):
                            eligible_types = self._eligible_types_for_mission(mission)
                            # Filter only types with available robots
                            candidates = []
                            for rtype in eligible_types:
                                if self.type_to_queue[rtype]:
                                    # Heuristic score: prefer more specialized types and those used less
                                    can_do_less = 1.0 / max(1, self._num_supported_missions_for_type(rtype))
                                    given = - float(self.type_missions_given[rtype])
                                    score = self.w1 * can_do_less + self.w2 * given
                                    candidates.append((score, rtype))

                            if not candidates:
                                continue

                            # Pick best type by score for this mission
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            _, best_type = candidates[0]
                            robot_id = self.type_to_queue[best_type].popleft()
                            selected = (idx, best_type, robot_id, mission)

                            break

                        if selected is None:
                            # No available robots for any mission in the queue yet
                            self.logger.debug("No eligible robots available for any mission; retrying")
                            time.sleep(0.03)
                            continue

                        # Finalize: remove mission at its index using deque rotation, update stats
                        idx, best_type, robot_id, mission = selected
                        if idx != 0:
                            # Rotate so that the target mission comes to the left
                            self.mission_queue.rotate(-idx)
                        # Remove the mission
                        self.mission_queue.popleft()
                        if idx != 0:
                            # Rotate back to original order
                            self.mission_queue.rotate(idx)

                        self.type_missions_given[best_type] += 1
                        self.next_pair = (robot_id, mission)
            time.sleep(0.01)



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

    @staticmethod
    def set_robot_dict(robot_dict):
        """Send robot_id -> robot_type mapping to queue server."""
        client = QueueClient()
        if not client.connect():
            return {"status": "error", "message": "Cannot connect to queue server"}
        resp = client._send_request({"action": "set_robot_dict", "robot_dict": robot_dict})
        client.disconnect()
        return resp
    
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
