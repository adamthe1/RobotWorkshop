import logging
import logging.handlers
import os
import sys
import glob
import socketserver
import struct
import pickle
import threading
import signal
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment variable to disable console output: LOG_NO_CONSOLE=1 or "true"
NO_CONSOLE = os.getenv('LOG_NO_CONSOLE', '0').lower() in ('1', 'true', 'yes')

class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            try:
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack('>L', chunk)[0]
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk = chunk + self.connection.recv(slen - len(chunk))
                obj = pickle.loads(chunk)
                record = logging.makeLogRecord(obj)
                self.handle_log_record(record)
            except Exception:
                break

    def handle_log_record(self, record):
        logger = logging.getLogger(record.name)
        logger.handle(record)

class LoggingServer:
    def __init__(self):
        self.host = os.getenv('LOGGING_HOST', 'localhost')
        self.port = int(os.getenv('LOGGING_PORT', 9020))
        
        self.server = None
        self.running = True
        self.setup_file_logging()

        

    def shutdown(self):
        """Clean shutdown of the logging server"""
        self.running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                print(f"Error shutting down logging server: {e}")
        sys.exit(0)

    def setup_file_logging(self):
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        log_filename = f"logs/autonomous_system.log"

        if os.path.isfile(log_filename):
            os.remove(log_filename)

        # Setup handlers
        file_handler = logging.FileHandler(log_filename, mode='w')
        handlers = [file_handler]
        if not NO_CONSOLE:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for h in handlers:
            h.setFormatter(formatter)
            root_logger.addHandler(h)
        
    def start(self):
        try:
            print(f"Starting logging server on {self.host}:{self.port}")
            self.server = socketserver.ThreadingTCPServer(
                (self.host, self.port), 
                LogRecordStreamHandler
            )
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("Logging server interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Logging server error: {e}")
        finally:
            self.shutdown()

class GlobalLogger:
    _instance = None
    _initialized = False
    
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            GlobalLogger._initialized = True
    
    def setup_logging(self):
        # Try to connect to logging server
        host = os.getenv('LOGGING_HOST', 'localhost')
        port = int(os.getenv('LOGGING_PORT', 9020))
        try:
            socket_handler = logging.handlers.SocketHandler(host, port)

            # Configure root logger with socket handler
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[socket_handler],
                force=True
            )
            
        except Exception:
            # Fallback to local file logging if server not available
            os.makedirs('logs', exist_ok=True)
            log_filename = f"logs/fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename, mode='w'),
                    logging.StreamHandler(sys.stdout)
                ],
                force=True
            )
        
        # Set specific loggers to appropriate levels
        logging.getLogger('RobotQueue').setLevel(logging.INFO)
        logging.getLogger('MainOrchestrator').setLevel(logging.INFO)
        logging.getLogger('CLI').setLevel(logging.INFO)
        logging.getLogger('BrainServer').setLevel(logging.INFO)
        logging.getLogger('BrainClient').setLevel(logging.INFO)
        
        logging.getLogger('MujocoServer').setLevel(logging.DEBUG)
        logging.getLogger('ActionManager').setLevel(logging.DEBUG)
        logging.getLogger('MujocoClient').setLevel(logging.INFO)
        logging.getLogger('PhysicsStateExtractor').setLevel(logging.INFO)
        logging.getLogger('RobotBodyControl').setLevel(logging.INFO)

    def get_logger(self, name):
        return logging.getLogger(name)


if __name__ == '__main__':
    server = LoggingServer()
    server.start()
else:
    # only client‐side processes get a socket handler
    global_logger = GlobalLogger()
    def get_logger(name):
        return global_logger.get_logger(name)