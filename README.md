# Autonomous_new

To run activate the virtual environment.
Install the requirements
Run python3 -m main_run


# env required

VISION_WEIGHTS_PATH = '/path/to/your/weights.pth'
MISSION_QUEUE_SIZE = 6  # Example size, adjust as needed

MUJOCO_HOST = 'localhost'
MUJOCO_PORT = 8600  # Default port for MuJoCo server

MAIN_DIRECTORY = "path/main"
USE_DUPLICATING_PATH = 1 # if to use the new duplicating mujoco
MUJOCO_MODEL_PATH = ${MAIN_DIRECTORY}/franka_emika_panda/scene_bar_new_ziv.xml
CONTROL_ROBOT_PATH = ${MAIN_DIRECTORY}/xml_robots/panda_scene.xml
LOAD_SAVED_STATE = 1
# franka_emika_panda/scene_bar_new_ziv.xml

# Robot configuration
FRANKA_PANDA_COUNT = 4
SO101_COUNT = 0


QUEUE_HOST = 'localhost'
QUEUE_PORT = 8700

LOGGING_HOST = 'localhost'
LOGGING_PORT = 8400
LOG_NO_CONSOLE = 1

NO_VIEWER = 0

BRAIN_HOST = 'localhost'
BRAIN_PORT = 8900
REPLAY_EPISODE_PATH=${MAIN_DIRECTORY}/finetuning/datasets/panda_teleop_dataset/data/chunk-000/episode_000001.parquet

OPENAI_API_KEY = api-key