# Finetuning and Teleoperation

This folder contains the teleoperation and recording tooling for creating mission datasets used by the Brain server to replay policies.

## Control Robot (SOLID)

- Entry point: `python3 finetuning/control_robot_solid.py`
- Purpose: Teleoperate a Franka Panda scene, record state/action sequences in a LeRobot-compatible parquet format, and optionally auto-load/save robot states.
- Scene: Set via `CONTROL_ROBOT_PATH` in `.env`. Defaults to `${MAIN_DIRECTORY}/xml_robots/panda_scene.xml` in the example env.

### Prerequisites
- Set `MAIN_DIRECTORY` in `.env` to your project root absolute path.
- Ensure the scene XML referenced by `CONTROL_ROBOT_PATH` exists.
- Optional GPU: set `USE_GPU_WSL=1` for WSL2 NVIDIA.

### How to Run
```
python3 finetuning/control_robot_solid.py
```

On start, the tool:
- Loads the XML scene
- Tries to auto-load the most recent saved state from `finetuning/saved_robot_states/*.npz` if present
- Prints available fixed cameras

### Controls
- XY: arrow keys 
- Z: `A` up, `D` down
- Gripper: `F` close, `G` open
- Save/Reset: `V` save, `B` reset to saved-or-xml, `Z` reset to XML
- Camera: `C` cycle among `free` and fixed cameras; mouse left-drag orbit, right-drag pan, wheel zoom
- IK toggle: `X`
- Stickiness toggle: `S` (sticky grasp capture/release)
- Joint jogging: numbers 1..7 increase, letters Q..U decrease (pairs: 1/Q, 2/W, 3/E, 4/R, 5/T, 6/Y, 7/U). Hold Shift for coarse steps.
- Recording: `M` start/stop segment, `N` finalize episode, `J` delete last recorded segment
- Exit: `ESC`

### Recording Output
- Format: LeRobot-style parquet with at least the `action` column; optionally `observation.state`.
- Default root: `${MAIN_DIRECTORY}/finetuning/datasets`.
- The recorder preserves actuator names order for later exact mapping during replay.

### Saved States
- Auto-load: newest file under `${MAIN_DIRECTORY}/finetuning/saved_robot_states/*.npz`
- Save now: press `V`
- Typical contents: `qpos`, `qvel`, optionally `joint_names`

## Mission Datasets

Policy replay expects mission datasets under:
```
finetuning/mission_episodes/<robot_type>/<mission>/<sub_mission>.parquet
```
- `<robot_type>`: keys from your runtime robot dict (e.g., `FrankaPanda`, `SO101`)
- `<mission>`: keys from `control_panel/missions.py:SUPPORTED_MISSIONS`
- `<sub_mission>`: file names matching each entry in that mission’s sub-mission list

Optional mapping: add `obs_to_action.json` next to parquet to map `observation.state` indices into actions.

Advanced replay envs (see root README for details): `REPLAY_SPEED`, `REPLAY_USE_OBS_AS_ACTION`, `REPLAY_OBS_FIRST_N`, `REPLAY_SAVED_STATE_DIR`, `RESET_HOLD_SECONDS`, `CONTROL_HZ`.

