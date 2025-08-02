# Control Panel: Mujoco Drinks Robot

## Overall Project Architecture
The project consists of:
- Training the Policy (model training, evaluation, dataset creation)
- Control Panel (CLI, mission manager, mission status, simulation loop)
- Mujoco Handler (scene loading, robot body control, physics extraction)
- The Brain (LLM for user input, embodiment manager, policy model inference)

**Data Flow:**
1. User interacts via CLI
2. Request processed by LLM
3. Mission parsed/decomposed by Control Panel
4. Sent to Mujoco Handler for execution
5. Status tracked/classified by vision model
6. Feedback loop to user

**Modules:**
- Control Panel: Coordinates missions and status
- Mujoco Handler: Loads scenes/configurations, controls robot
- The Brain: Processes input, manages embodiment and policy inference
- Training: Manages policy/model creation, evaluation, and datasets

## 1. Overview and Purpose
The Control Panel coordinates between user input (via CLI/LLM), simulation logic, and mission state for a Mujoco-based robot arm system that pours drinks. It parses and manages requests, decomposes missions, interfaces with simulation, and monitors progress. 

## 2. CLI Design
- Invoked on simulation loop start
- Prompts user for requested drink
- Forwards request to LLM (OpenAI API format)
- Displays current mission and subtask status

## 3. Mission Manager
- Lists supported missions (trained drink tasks)
- Parses LLM input into valid missions/submissions
- Dispatches parsed missions to simulation controller

## 4. Mission Status Tracking
- Receives simulation/scene feedback
- Invokes small vision model to classify subtask completion
- Reports and resets status for ongoing and new missions

## 5. Queue
- A queue is implemented to coordinate missions across robots

## 5. Control Panel API
- Entry points: start mission, update status, reset
- Connectors for LLM API, Mujoco simulation calls, vision model API

## 6. Integration & Data Flow
- CLI → LLM → mission manager → simulation → status tracker → user
- Details on connections to other modules: Mujoco handler, The Brain

- This project has been designed with SOLID principles, we need to always try and keep to these rules especially when adding new functions and classes


