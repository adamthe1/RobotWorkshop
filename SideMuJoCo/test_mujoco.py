#!/usr/bin/env python3
"""
Client test for MuJoCoServer with debug logging.
Assumes the server is already running on localhost:5555.
This script will:
  1) In an infinite loop: request the current state, send a dummy action (zeros), and step the simulation.

Debug prints and socket timeouts help diagnose hangs.
"""
import socket
import pickle
import struct
import time
import traceback
from packet_example import Packet
import numpy as np

HOST = 'localhost'
PORT = 5555


def send_and_recv(sock, packet):
    """
    Send a pickled packet and receive the pickled reply, with debug logs.
    """
    try:
        print(f"DEBUG: send_and_recv - sending packet: {packet}")
        data_out = pickle.dumps(packet)
        sock.sendall(struct.pack('!I', len(data_out)) + data_out)

        # Read length prefix
        size_data = sock.recv(4)
        if not size_data:
            raise ConnectionError("No data received for length prefix")
        size = struct.unpack('!I', size_data)[0]
        print(f"DEBUG: send_and_recv - expecting {size} bytes reply")

        # Read the full reply
        buf = b''
        while len(buf) < size:
            chunk = sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Incomplete payload: connection closed")
            buf += chunk
        reply = pickle.loads(buf)
        print(f"DEBUG: send_and_recv - received reply: {reply}")
        return reply
    except Exception:
        print("ERROR: Exception in send_and_recv:\n" + traceback.format_exc())
        raise


def main():
    print("DEBUG: main - sleeping before connect...")
    time.sleep(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        try:
            sock.connect((HOST, PORT))
            print(f"Connected to server at {HOST}:{PORT}")
        except Exception:
            print("ERROR: Failed to connect to server:\n" + traceback.format_exc())
            return

        try:
            while True:
                # 1) Request state via Packet
                print("DEBUG: main - requesting state...")
                state=send_and_recv(sock, Packet(robot_id="r1"))
                print("DEBUG: main - received state:", state)
                print(f"State → qname={state.joint_names} qpos={state.qpos}, qvel={state.qvel}, time={state.time} images={state.images}")

                # 2) Send dummy action via Packet
                qpos = state.qpos or np.array([])
                dummy_action = np.zeros_like(qpos)
                print("DEBUG: main - sending dummy action:", dummy_action)
                ack: Packet = send_and_recv( sock,
                    Packet(robot_id="r1",
                        action=dummy_action) )
                print("Action ack status:", ack)  # or however your server encodes the ack

                # Pause between steps to control speed
                time.sleep(2)

        except KeyboardInterrupt:
            print("DEBUG: main - interrupted by user, exiting loop")
        except Exception:
            print("ERROR: Exception in main loop:\n" + traceback.format_exc())
        finally:
            print("DEBUG: main - closing socket")
            sock.close()


if __name__ == '__main__':
    main()