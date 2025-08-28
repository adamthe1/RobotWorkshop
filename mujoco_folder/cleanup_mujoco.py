#!/usr/bin/env python3
# filepath: /home/adam/Documents/coding/autonomous/check_ports.py

import socket
import subprocess
import sys
import os
import signal
from dotenv import load_dotenv

load_dotenv()

def check_port(host, port):
    """Check if a port is in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0  # True if port is open/in use
    except:
        return False

def get_process_using_port(port):
    """Get the process ID using a specific port"""
    try:
        # Use netstat to find process using the port
        result = subprocess.run(
            ['netstat', '-tlnp'], 
            capture_output=True, 
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if f':{port} ' in line and 'LISTEN' in line:
                parts = line.split()
                if len(parts) >= 7:
                    pid_program = parts[6]
                    if '/' in pid_program:
                        pid = pid_program.split('/')[0]
                        program = pid_program.split('/')[1]
                        return pid, program
        return None, None
    except Exception as e:
        print(f"Error getting process for port {port}: {e}")
        return None, None

def kill_process(pid, name=""):
    """Kill a process by PID"""
    try:
        pid = int(pid)
        os.kill(pid, signal.SIGTERM)
        print(f"✓ Killed process {pid} ({name})")
        return True
    except ProcessLookupError:
        print(f"✗ Process {pid} ({name}) not found")
        return False
    except PermissionError:
        print(f"✗ Permission denied to kill process {pid} ({name})")
        return False
    except Exception as e:
        print(f"✗ Error killing process {pid} ({name}): {e}")
        return False

def main():
    # Define the ports your system uses
    ports_to_check = {
        'MuJoCo Server': int(os.getenv('MUJOCO_PORT', 8600)),
        'Queue Server': int(os.getenv('QUEUE_PORT', 8700)), 
        'Brain Server': int(os.getenv('BRAIN_PORT', 8900)),
        'Logging Server': int(os.getenv('LOGGING_PORT', 8800))
    }
    
    host = 'localhost'
    
    print("🔍 Checking port usage...")
    print("=" * 50)
    
    processes_to_kill = []
    
    for service_name, port in ports_to_check.items():
        is_open = check_port(host, port)
        
        if is_open:
            pid, program = get_process_using_port(port)
            print(f"🔴 {service_name:15} | Port {port:5} | IN USE    | PID: {pid} ({program})")
            if pid:
                processes_to_kill.append((pid, program, service_name))
        else:
            print(f"🟢 {service_name:15} | Port {port:5} | FREE")
    
    # Check for any python processes that might be related
    print("\n🔍 Checking for related Python processes...")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'autonomous'], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            related_pids = result.stdout.strip().split('\n')
            for pid in related_pids:
                if pid:
                    # Get process info
                    try:
                        cmd_result = subprocess.run(
                            ['ps', '-p', pid, '-o', 'cmd='], 
                            capture_output=True, 
                            text=True
                        )
                        cmd = cmd_result.stdout.strip()
                        print(f"🔴 Related Process | PID: {pid:5} | {cmd}")
                        processes_to_kill.append((pid, 'python', 'Related'))
                    except:
                        pass
    except:
        pass
    
    # Ask user if they want to kill processes
    if processes_to_kill:
        print(f"\n📝 Found {len(processes_to_kill)} processes to clean up")
        print("=" * 50)
        
        if len(sys.argv) > 1 and sys.argv[1] == '--kill':
            # Auto-kill mode
            print("🗡️  Auto-killing all processes...")
            for pid, program, service in processes_to_kill:
                kill_process(pid, f"{service}/{program}")
        else:
            # Interactive mode
            response = input("Kill all these processes? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n🗡️  Killing processes...")
                for pid, program, service in processes_to_kill:
                    kill_process(pid, f"{service}/{program}")
            else:
                print("✋ Skipped killing processes")
                
        # Re-check ports after cleanup
        print("\n🔍 Re-checking ports after cleanup...")
        print("=" * 50)
        for service_name, port in ports_to_check.items():
            is_open = check_port(host, port)
            status = "IN USE" if is_open else "FREE"
            icon = "🔴" if is_open else "🟢"
            print(f"{icon} {service_name:15} | Port {port:5} | {status}")
    else:
        print("\n✅ All ports are free!")

if __name__ == '__main__':
    main()