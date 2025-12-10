
## Structure of the Framework

Sensor data are transmitted from the MATRiX simulation environment to a ROS2 bridging module, which forwards the received data to the agent process via ZMQ.

The agent runs inside a conda environment and drives LLM/VLM models through HuggingFace models, Ollama, or cloud-based APIs. HuggingFace models are served by a local inference server running in a separate conda environment.

Both agents and model backends are pluggable through environment-level isolation (per-component conda environments), enabling flexible deployment and easy switching of model providers.

## Brief Description of the Baseline Algorithm:
The agent receives RGB, depth and pose which are transmitted by the bridging module from MATRiX simulation environment. Based on this information, point clouds are back-projected to the space and then projected to the ground to create an occupancy map, which separates free space from obstacles. Frontiers, which are the boundaries between explored and unexplored regions are extracted and used as candidate exploration goals.

In parallel, a scene graph is built by prompting the VLM to infer spatial relationships among detected objects. The scene graph provides a semantic prior over the environment: frontiers located in regions whose surrounding objects or room context align with the goal graph receive higher scores, while semantically irrelevant frontiers are down-weighted. Combined with distance-based weighting, the frontier with the highest overall score is selected as the navigation target.

After selecting a goal frontier, a Fast Marching Method (FMM) planner computes a collision-free path on the occupancy map. At each timestep, the agent executes a short-horizon action—turning or moving forward—toward the next waypoint while continuously updating the local map and its pose. During navigation, candidate objects are validated either through depth-aware geometric consistency and local feature matching (for image-based goals) or through semantic matching using the VLM and the scene graph (for text-based goals). Once the agent arrives within a threshold distance and the detected object or semantic description aligns with the specified goal, the episode is terminated successfully.

## Guide

### 1. Follow Step by Step Procedure in [Setup](https://github.com/zsibot/zsibot_vln/blob/dev/README.md#%EF%B8%8F-setup) 

### 2. Try different LLMs by Configuring `configs/config_matrix.yaml`

Example huggingface configuration:

```bash
cloud_api: False
api_key: "huggingface"
base_url: "http://localhost:8000/v1"
llm_model: "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
vlm_model: "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
```

Example ollama configuration:

```bash
cloud_api: False
api_key: "ollama"

base_url: "http://localhost:11434/v1/" # default local ollama server
#or
#base_url: "http://192.160.xx.xx:11434/v1/" # any ollama server

llm_model: "gemma3:4b" #
vlm_model: "gemma3:4b" # or any other model supported by ollama
```

Example cloud api configuration:

```bash
cloud_api: True
api_key: "" # for safety reason, this comes from environment variable processed in main.py
            # just do this on shell: export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'
base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm_model: "qwen3-vl-flash"
vlm_model: "qwen3-vl-flash" #or qwen3-vl-plus and any other supported model
```

## Develope Your Own VLN Algorithm

ZsiBot is designed as a framework which provides interfaces for connecting the MATRiX simulator with VLN algorithms. Developers are welcome to add new VLN algorithms under `zsibot_vln/agents/zeroshot/` or `zsibot_vln/agents/finetune/`. The key integration points are summarized below.

### Configuring the VLN Baseline

See the comments and descriptions in configuration file `zsibot/configs/config_matrix.yaml`.

### Configuring Matrix sensors
It is important to understand `zsibot_vln/configs/config.json`, which configures the robot in MATRiX. After editing, this file should be copied into the MATRiX directory for your development.

```bash
cp zsibot_vln/configs/config.json matrix/config/
```

The most important settings are explained in the comments.

```bash
{
    "robot": {
        "robot_type": "xgb",
        "weapon": "",
        "position": {
            "x": 0.0,   # wake up position relative to the default one
            "y": 0.0,
            "z": 0.0
        },
        "rotation": {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        },
        "synchronous_mode": false,
        "sensors": {
            "camera": {  #camera pose relative to robot body center
                "position": {
                    "x": 17.0,
                    "y": 0.0,
                    "z": 1.0
                },
                "rotation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                },
                "height": 1080, #resolution
                "width": 1920,
                "sensor_type": "rgb",
                "topic": "/image_raw/compressed", #ros2 topic
                "fov": 90.0,
                "frequency": 10.0
            },
            "depth_sensor": { # depth added on top of the camera image
                "position": {
                    "x": 17.0,
                    "y": 0.0,
                    "z": 1.0
                },
                "rotation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                },
                "height": 1080, # must be the same to camera resolution
                "width": 1920,
                "sensor_type": "depth2", #depth2 is the right sensor for vln task, the measurement range is 0 to 10 meters.
                "topic": "/image_raw/compressed/depth",
                "fov": 90.0,
                "frequency": 10.0
            },
            "lidar": { # lidar sensor which is not used for the current VLN baseline model
                "position": {
                    "x": 13.011,
                    "y": 2.329,
                    "z": 17.598
                },
                "rotation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                },
                "sensor_type": "mid360",
                "topic": "/livox/lidar",
                "draw_points": false,
                "random_scan": false,
                "frequency": 10.0
            }
        }
    }
}

```

### Bridge module
Located at `zisibot/bridge`

This is a bidirectional bridging module. You can switch between transmitting all sensor data or only the data required by the baseline model by toggling the `transfer_all` variable in `bridge/src/env_bridge/env_bridge/env_bridge.py`. The directory `bridge/src/mc_sdk_bridge/` contains the program that receives high-level commands (forward, left, right, stop) and translates them into low-level control commands for the robot.

### Environment wrapper
Located at `env_matrix.py`

Sensor data are received, processed, and returned to the agent. High-level commands produced by the agent are sent to the bridge, which forwards them to the robot in MATRiX. Actions are only sent after the robot’s state is detected to be stable.

### Main entry point
`main.py` is the entry point for selecting and running a VLN agent.

### Utility functions
Located at `zsibot/utils`

Utility functions are provided here, including a visualization tool.
