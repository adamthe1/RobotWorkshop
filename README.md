# Autonomous Robot Bartender

A multi-robot simulation and control system built with MuJoCo, supporting teleoperation, recording, and policy replay.

<img src="example_gifs/pour_beer.gif" width="80%" height="80%"/>

## Quick Start

1. **Activate virtual environment** and install requirements  
   `source venv/bin/activate |
   pip install -r requirements.txt`
2. **Set up environment variables** (copy .exampleenv -> .env and add api-key, maindir)
    `cp .exampleenv .env`
3. **Run simulation**: `python3 -m main_run`
4. **Talk to Bartender**: Ask for a drink
5. **Quit**: Press CTRL+C in terminal to quit

## System Architecture

- **MuJoCo Server**: Physics simulation and robot control
- **Brain Server**: Policy inference and action generation
- **Queue Server**: Mission and task management
- **Logging Server**: Centralized logging across all components
- **Viewer**: Real-time 3D visualization

## Features

- **Multi-robot simulation** with Franka Panda and SO101 robots
- **Real-time teleoperation** and recording
- **Policy replay** from recorded episodes
- **GPU acceleration** support for WSL2
- **Distributed architecture** with separate brain, logging, and queue servers

## Current Supported Drinks
- **Franka Panda**: "pour beer", "pour mojito"
## Configuration

Create a `.env` file in the project root with the following variables:

### Core System Settings
```bash
# Main project directory (absolute path)
MAIN_DIRECTORY = "/home/user/Documents/coding/autonomous"

# Recommended to run full bar. Will display default menu if not configured
OPENAI_API_KEY = api_key
GLM_API_KEY = api_key

# Use GPU (1) or CPU (0)
USE_GPU_WSL = 1

# How many Robots of each Type you want
FRANKA_PANDA_COUNT = 1
SO101_COUNT = 0

```
**See .exampleenv for other setting**


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
 Place mission datasets under `finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet`.


### Multiple Robots
Increase `FRANKA_PANDA_COUNT` and `SO101_COUNT` to add more robots. GPU rendering is recommended for 3+ robots.

## Adding Missions

- Define missions and sub-missions in `control_panel/missions.py` (`SUPPORTED_MISSIONS`) and assign per robot in `SUPPORTED_MISSIONS_PER_ROBOT`.
- Provide datasets under `finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet` where `<sub_mission>` file names match each entry you added.

