# Autonomous Robotics System

A multi-robot simulation and control system built with MuJoCo, supporting teleoperation, recording, and policy replay.

## Quick Start

1. **Activate virtual environment** and install requirements
2. **Set up environment variables** (copy .exampleenv -> .env and add api-key, maindir)
3. **Run simulation**: `python3 -m main_run`
4. **Record teleoperation data**: `python3 finetuning/control_robot_solid.py`

## Features

- **Multi-robot simulation** with Franka Panda and SO101 robots
- **Real-time teleoperation** and recording
- **Policy replay** from recorded episodes
- **GPU acceleration** support for WSL2
- **Distributed architecture** with separate brain, logging, and queue servers

## Configuration

Create a `.env` file in the project root with the following variables:

### Core System Settings
```bash
# Main project directory (absolute path)
MAIN_DIRECTORY = "/home/user/Documents/coding/autonomous"

# GPU rendering (recommended for WSL2 with NVIDIA)
USE_GPU_WSL = 1

# Disable console logging (recommended 1)
LOG_NO_CONSOLE = 1

# Disable viewer for headless operation
NO_VIEWER = 0
```

### Robot Configuration
```bash
# Use dynamic scene generation (recommended)
USE_DUPLICATING_PATH = 1

# Number of robots per type
FRANKA_PANDA_COUNT = 1
SO101_COUNT = 2

# Load saved simulation state for the robots
LOAD_SAVED_STATE = 1
```

### Model Paths
```bash
# Main scene file (when not using duplicating path)
MUJOCO_MODEL_PATH = ${MAIN_DIRECTORY}/franka_emika_panda/scene_bar_new_ziv.xml

# Which scene to generate when recording the episodes
CONTROL_ROBOT_PATH = ${MAIN_DIRECTORY}/xml_robots/panda_scene.xml
```

### Network Configuration
```bash
# MuJoCo physics server
MUJOCO_HOST = 'localhost'
MUJOCO_PORT = 8600

# Mission queue server
QUEUE_HOST = 'localhost'
QUEUE_PORT = 8700

# Logging server
LOGGING_HOST = 'localhost'
LOGGING_PORT = 8400

# Brain/policy server
BRAIN_HOST = 'localhost'
BRAIN_PORT = 8900
```

### Recording & Replay
```bash
# Episode replay file for policy testing
REPLAY_EPISODE_PATH = ${MAIN_DIRECTORY}/finetuning/datasets/panda_teleop_dataset/data/chunk-000/episode_000001.parquet

# Mission queue size
MISSION_QUEUE_SIZE = 6
```

### Needed models
```bash
# Vision model weights for checking mission status
VISION_WEIGHTS_PATH = '/path/to/your/weights.pth'

# OpenAI API key (if using language models)
OPENAI_API_KEY = 'your-api-key-here'
```

## Usage

### Running the Simulation
```bash
# Start the full system (after starting venv)
python3 -m main_run
```

This launches:
- MuJoCo physics server
- Brain server (for policy inference)
- Mission queue server
- Logging server
- 3D viewer (if enabled)

### Recording Teleoperation Data
```bash
# Start teleoperation recording
python3 finetuning/control_robot_solid.py
```

This allows you to:
- Control robots via keyboard
- Record state-action pairs
- Save data in LeRobot-compatible format

### Replaying Policies
Set `REPLAY_EPISODE_PATH` to your recorded episode file and run the simulation. The brain server will replay actions from the episode.

## System Architecture

- **MuJoCo Server**: Physics simulation and robot control
- **Brain Server**: Policy inference and action generation
- **Queue Server**: Mission and task management
- **Logging Server**: Centralized logging across all components
- **Viewer**: Real-time 3D visualization

## Troubleshooting

### GPU Rendering Issues
- Ensure NVIDIA drivers are installed in WSL2
- Set `USE_GPU_WSL = 1` for hardware acceleration
- Check `nvidia-smi` works in WSL2

### Port Conflicts
The system automatically detects and resolves port conflicts by offering to kill conflicting processes.

### Multiple Robots
Increase `FRANKA_PANDA_COUNT` and `SO101_COUNT` to add more robots. GPU rendering is recommended for 3+ robots.