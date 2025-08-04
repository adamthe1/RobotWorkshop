import logging
import os
import sys
import glob
from datetime import datetime

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
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        log_filename = f"logs/autonomous_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Clear any existing log files with same pattern in logs directory
        for old_log in glob.glob("logs/autonomous_system_*.log"):
            try:
                os.remove(old_log)
            except OSError:
                pass
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Override any existing configuration
        )
        
        # Set specific loggers to appropriate levels
        logging.getLogger('RobotQueue').setLevel(logging.INFO)
        logging.getLogger('MainOrchestrator').setLevel(logging.INFO)
        logging.getLogger('CLI').setLevel(logging.INFO)
        logging.getLogger('MissionManager').setLevel(logging.INFO)
        logging.getLogger('MujocoServer').setLevel(logging.INFO)
        logging.getLogger('MujocoClient').setLevel(logging.INFO)

    def get_logger(self, name):
        return logging.getLogger(name)

# Global instance
global_logger = GlobalLogger()

def get_logger(name):
    """Get a logger with the specified name"""
    return global_logger.get_logger(name)