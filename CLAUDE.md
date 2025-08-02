# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a autonomous robot bartender system that uses MuJoCo physics simulation to control a Franka Emika Panda robot arm for pouring drinks. The system coordinates multiple components through a main orchestrator and uses LLM-based natural language processing for user interactions.

## System Architecture

### Core Components

1. **Main Orchestrator** (`main_run.py`): Central coordinator that manages all system components
2. **MuJoCo Server** (`mujoco_server.py`): Physics simulation server for robot control
3. **Control Panel** (`control_panel/`): Mission management and user interface
4. **Robot Queue** (`control_panel/robot_queue.py`): Thread-safe mission queuing system
5. **Mission Manager** (`control_panel/mission_manager.py`): Handles mission decomposition and execution
6. **LLM Client** (`control_panel/LLM.py`): Interfaces with language models for natural language processing
7. **Global Logger** (`logger_config.py`): Centralized logging system

### Data Flow

1. User interacts through CLI (`run.py`)
2. LLM processes natural language input and extracts drink orders
3. Orders are queued in the RobotQueue system
4. Main orchestrator dequeues missions and coordinates execution
5. Mission manager breaks down missions into subtasks
6. Actions are sent to MuJoCo server for physics simulation
7. System logs all activities through global logger

## Key Commands

### Running the System

```bash
# Start the complete system
python main_run.py

# Start the CLI interface directly
python run.py

# Start MuJoCo server independently
python mujoco_server.py

# Test MuJoCo server connection
python test_mujoco.py
```

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual tests
python test_mujoco.py
```

## Module Structure

### Control Panel Module (`control_panel/`)

- **`LLM.py`**: LLM client for natural language processing (supports OpenAI and custom backends)
- **`mission_manager.py`**: Mission decomposition and state management
- **`robot_queue.py`**: Thread-safe mission queuing for multiple robots
- **`mission_status.py`**: Vision-based mission completion tracking
- **`missions.py`**: Defines supported missions and subtasks
- **`system_prompt.py`**: System prompt for LLM interactions
- **`args.py`**: Configuration arguments and paths

### Communication Protocol

The system uses a packet-based communication protocol over TCP sockets:

- **`Packet` class** (`packet_example.py`): Data structure for robot state and actions
- **TCP socket communication**: Length-prefixed pickled packets
- **Default port**: 5555

### Configuration

- **Model path**: `/home/adam/Documents/coding/autonomous/franka_emika_panda/mjx_scene.xml`
- **Server host**: `localhost`
- **Server port**: `5555`
- **Logging**: Logs to `logs/autonomous_system_YYYYMMDD_HHMMSS.log`

## Development Guidelines

### Code Style

- Follow SOLID principles throughout the codebase
- Use type hints where appropriate
- Maintain thread safety in queue operations
- Use the global logging system for consistent logging

### Key Classes and Their Roles

1. **`MainOrchestrator`**: System lifecycle management, coordinates all components
2. **`MujocoClient`**: TCP client for MuJoCo server communication
3. **`RobotQueue`**: Thread-safe mission queuing with robot-specific locks
4. **`MissionManager`**: Mission decomposition and state tracking
5. **`LLMClient`**: Natural language processing for drink orders
6. **`MuJoCoServer`**: Physics simulation and robot control

### Mission Structure

Missions are defined in `control_panel/missions.py` as hierarchical structures:
- **Mission**: High-level task (e.g., "pour beer")
- **Submissions**: Individual steps (e.g., "pick up yellow bottle", "pour bottle into cup")

### Logging

The system uses a centralized logging configuration:
- Log files are created in the `logs/` directory
- Each run creates a new timestamped log file
- Loggers are available for different components (RobotQueue, MainOrchestrator, CLI, etc.)

### Testing

- `test_mujoco.py`: Integration test for MuJoCo server communication
- Use this script to verify server connectivity and basic functionality

## Environment Setup

The system requires:
- Python 3.10+
- MuJoCo physics engine
- numpy for numerical operations
- requests for HTTP communication
- typer for CLI interface
- pillow for image processing

## Important Notes

- The system is designed to run with multiple robots, but currently supports `robot_1`
- Mission queue has a maximum job limit (currently 1 job per robot)
- The system uses dummy actions for policy inference (placeholder for future ML model integration)
- All network communication uses pickle serialization - ensure security in production environments