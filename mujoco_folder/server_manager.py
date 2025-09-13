#!/usr/bin/env python3

import subprocess
import sys
import time
import signal
import psutil
import os
from pathlib import Path

# Add the project root to Python path so we can import modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from logger_config import get_logger
from dotenv import load_dotenv

load_dotenv()

class ServerManager:
    def __init__(self):
        self.logger = get_logger('ServerManager')
        self.processes = {}
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, _signum, _frame):
        self.logger.info("Shutdown signal received")
        self.kill_all_servers()
        
    def start_logging_server(self):
        """Start the logging server"""
        if 'logging' in self.processes:
            self.logger.warning("Logging server already running")
            return
            
        self.logger.info("Starting Logging server...")
        project_root = Path(__file__).resolve().parent.parent  # Go up to RobotWorkshop
        
        try:
            process = subprocess.Popen([
                sys.executable,
                "-m", "logger_config"
            ], cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)})
            
            self.processes['logging'] = process
            time.sleep(1)  # Give server time to start
            self.logger.info(f"Logging server started with PID: {process.pid}")
        except Exception as e:
            self.logger.error(f"Failed to start logging server: {e}")

    def start_mujoco_server(self):
        """Start the MuJoCo server"""
        if 'mujoco' in self.processes:
            self.logger.warning("MuJoCo server already running")
            return
            
        self.logger.info("Starting MuJoCo server...")
        project_root = Path(__file__).resolve().parent.parent
        
        try:
            process = subprocess.Popen([
                sys.executable,
                "-m", "mujoco_folder.mujoco_server_merge"
            ], cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)})
            
            self.processes['mujoco'] = process
            time.sleep(1)
            self.logger.info(f"MuJoCo server started with PID: {process.pid}")
        except Exception as e:
            self.logger.error(f"Failed to start MuJoCo server: {e}")

    def start_queue_server(self):
        """Start the queue server"""
        if 'queue' in self.processes:
            self.logger.warning("Queue server already running")
            return
            
        self.logger.info("Starting Queue server...")
        project_root = Path(__file__).resolve().parent.parent
        
        try:
            process = subprocess.Popen([
                sys.executable,
                "-m", "control_panel.robot_queue_locks"
            ], cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)})
            
            self.processes['queue'] = process
            time.sleep(1)
            self.logger.info(f"Queue server started with PID: {process.pid}")
        except Exception as e:
            self.logger.error(f"Failed to start queue server: {e}")

    def start_brain_server(self):
        """Start the brain server"""
        if 'brain' in self.processes:
            self.logger.warning("Brain server already running")
            return
            
        self.logger.info("Starting Brain server...")
        project_root = Path(__file__).resolve().parent.parent
        
        try:
            process = subprocess.Popen([
                sys.executable,
                "-m", "brain.brain_server"
            ], cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)})
            
            self.processes['brain'] = process
            time.sleep(1)
            self.logger.info(f"Brain server started with PID: {process.pid}")
        except Exception as e:
            self.logger.error(f"Failed to start brain server: {e}")

    def start_cli(self):
        """Start the CLI interface"""
        if 'cli' in self.processes:
            self.logger.warning("CLI already running")
            return
            
        self.logger.info("Starting CLI...")
        project_root = Path(__file__).resolve().parent.parent
        
        try:
            # Build CLI command: run main CLI (no subcommand)
            cmd = [
                sys.executable,
                "-m", "brain.CLI.run_cli",
            ]
            # Optionally enable voice mode via env
            if os.getenv('CLI_VOICE', '0').lower() in ('1', 'true', 'yes'):
                cmd.append("--voice")

            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env={**os.environ, "PYTHONPATH": str(project_root)}
            )
            
            self.processes['cli'] = process
            self.logger.info(f"CLI started with PID: {process.pid}")
        except Exception as e:
            self.logger.error(f"Failed to start CLI: {e}")

    def start_all_servers(self):
        """Start all servers in the correct order"""
        self.logger.info("Starting all servers...")
        
        # Start servers in dependency order
        self.start_logging_server()
        self.start_mujoco_server()
        self.start_queue_server()
        self.start_brain_server()
        
        # Wait a bit for all servers to stabilize
        time.sleep(2)
        self.logger.info("All servers started")

    def kill_process_tree(self, pid):
        """Kill a process and all its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.kill()
                    self.logger.debug(f"Killed child process {child.pid}")
                except psutil.NoSuchProcess:
                    pass
            
            # Kill parent
            parent.kill()
            self.logger.debug(f"Killed parent process {pid}")
            
            # Wait for processes to die
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            
            # Force kill any remaining processes
            for proc in alive:
                try:
                    proc.kill()
                    self.logger.warning(f"Force killed stubborn process {proc.pid}")
                except psutil.NoSuchProcess:
                    pass
                    
        except psutil.NoSuchProcess:
            self.logger.debug(f"Process {pid} already terminated")
        except Exception as e:
            self.logger.error(f"Error killing process tree {pid}: {e}")

    def kill_server(self, server_name):
        """Kill a specific server"""
        if server_name not in self.processes:
            self.logger.warning(f"Server '{server_name}' not found in process list")
            return
            
        process = self.processes[server_name]
        
        if process.poll() is not None:
            self.logger.info(f"Server '{server_name}' already terminated")
            del self.processes[server_name]
            return
            
        self.logger.info(f"Killing {server_name} server (PID: {process.pid})...")
        
        try:
            # First try graceful termination
            process.terminate()
            try:
                process.wait(timeout=5)
                self.logger.info(f"{server_name} server terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill the entire process tree
                self.logger.warning(f"{server_name} server didn't terminate gracefully, force killing...")
                self.kill_process_tree(process.pid)
                
        except Exception as e:
            self.logger.error(f"Error killing {server_name} server: {e}")
            # Try to force kill anyway
            try:
                self.kill_process_tree(process.pid)
            except:
                pass
                
        # Remove from process list
        del self.processes[server_name]

    def kill_all_servers(self):
        """Kill all running servers and ensure they're terminated"""
        self.running = False
        self.logger.info("Shutting down all servers...")
        
        # Kill in reverse dependency order
        server_order = ['cli', 'brain', 'queue', 'mujoco', 'logging']
        
        for server_name in server_order:
            if server_name in self.processes:
                self.kill_server(server_name)
        
        # Wait a moment for cleanup
        time.sleep(1)
        
        # Double-check: kill any remaining processes by name
        self._cleanup_by_process_name()
        
        self.logger.info("All servers shut down")

    def _cleanup_by_process_name(self):
        """Cleanup any remaining processes by searching for known process names"""
        process_patterns = [
            'mujoco_server_merge',
            'robot_queue_locks',
            'brain_server',
            'run_cli',
            'logger_config'
        ]
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                for pattern in process_patterns:
                    if pattern in cmdline and 'python' in cmdline.lower():
                        self.logger.warning(f"Found orphaned process: {proc.info['pid']} - {cmdline}")
                        try:
                            self.kill_process_tree(proc.info['pid'])
                        except:
                            pass
                        break
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def get_server_status(self):
        """Get status of all servers"""
        status = {}
        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = f"Running (PID: {process.pid})"
            else:
                status[name] = "Terminated"
        return status

    def wait_for_servers(self):
        """Wait for all servers and monitor their status"""
        try:
            while self.running and self.processes:
                # Check if any process has died
                dead_processes = []
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        self.logger.warning(f"{name} server terminated unexpectedly")
                        dead_processes.append(name)
                
                # Remove dead processes
                for name in dead_processes:
                    del self.processes[name]
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.kill_all_servers()

def main():
    """Main function for standalone usage"""
    manager = ServerManager()
    
    try:
        manager.start_all_servers()
        print("All servers started. Press Ctrl+C to stop.")
        manager.wait_for_servers()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.kill_all_servers()
        print("Cleanup complete")

if __name__ == '__main__':
    main()
