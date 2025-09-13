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
GLM_API_KEY = 'your-api-key-here'
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

- See `finetuning/README.md` for full teleop controls, recording workflow, saved-state behavior, and dataset details.

This allows you to:
- Control robots via keyboard
- Record state-action pairs
- Save data in LeRobot-compatible format

### Replaying Policies
Preferred: place mission datasets under `finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet`.
Legacy: you can set `REPLAY_EPISODE_PATH` to a single episode parquet to test policy replay.

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

## Adding Missions

- Define missions and sub-missions in `control_panel/missions.py` (`SUPPORTED_MISSIONS`) and assign per robot in `SUPPORTED_MISSIONS_PER_ROBOT`.
- Provide datasets under `finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet` where `<sub_mission>` file names match each entry you added.
- The replayer loads actions exactly as recorded. If a reset step like `reset_before_new_mission` has no parquet, a synthetic reset sequence is generated and held for a configurable duration.
- Optional: add `obs_to_action.json` alongside parquet to map `observation.state` indices into actions when `REPLAY_USE_OBS_AS_ACTION=1`.

## Additional Environment Variables

Beyond the core settings above, these control replay and state handling (add to `.env` as needed):
- CONTROL_HZ: Control/replay loop frequency in Hz (default 60).