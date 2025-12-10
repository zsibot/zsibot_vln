
# ZsiBot VLN

ZsiBot VLN provides a framework for developing, testing, and deploying Vision-Language Navigation (VLN) algorithms, unifying the [MATRiX](https://github.com/zsibot/matrix) simulation platform and VLN algorithms into a single, extensible pipeline. It also includes a zeroshot VLN baseline model, which serves as both a reference implementation and a practical starting point for research or product development.

➡️ **[Full Development Guide](./docs/guide.md)**

<p align="center">
  <img src="assets/vln.gif" width="480">
</p>

## 🗂️ Project Structure

```text
zsibot_vln/
├── agents/
│ └── zeroshot/
│     └── unigoal/  # baseline VLN model
│ └── finetune/     # todo
├── assets/
├── bridge/
│ └── src/          # ROS2 ↔ ZMQ bidirectional bridging module
├── configs/        # configuration files
├── docs/
├── envs/           # MATRiX environment, adaptable to real-world robots
├── goals/          # example image goals
├── llms/           # prepared LLM/VLM HuggingFace models
├── outputs/
├── third_party/
├── main.py
├── requirements.txt
└── README.md
```

## 🛠️ Setup

### Prerequisites
* CUDA-capable GPU (Nvidia 4090 recommended when using local LLMs)
* A gentle recommendation: use VPN for Git and model weight downloads


### Clone Repository and Install zmq3

```bash
git clone https://github.com/zsibot/zsibot_llm.git
sudo apt install libzmq3-dev
```

### Install the Simulator (MATRiX)

Follow the [MATRiX](https://github.com/zsibot/matrix) installation instructions and then:

```bash
#Update matrix/config.json
cp zsibot_vln/configs/config.json matrix/config/
```

### Prepare LLM Access

Option 1: Huggingface
```bash
conda create -n smol python=3.9 -y
conda activate smol
conda install --freeze-installed -c nvidia cuda-toolkit=12.4 -y
conda install --freeze-installed pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c conda-forge libstdcxx-ng
pip install -U transformers datasets evaluate accelerate timm
pip install num2words fastapi uvicorn hf_xet
pip install --no-cache-dir --no-build-isolation --verbose flash-attn
# download weights:
python zsibot_llm/llms/smolvlm2_256m_video_instruct/smolvlm2_256m_video_instruct.py
```

Option 2: Cloud LLM/VLM API (fetch API-key from e.g. [Aliyun Bailian](https://bailian.console.aliyun.com/))
```bash
export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'
```

Option 3: Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b # or any other models from ollama
```

### Install VLN Baseline Model

Install and run the baseline following the instructions:

➡️ **[VLN Baseline Installation Guide](./docs/vln_baseline.md)**



## 🚀 Run

```bash
#run local server on shell 0 (can be sikpped when using ollama or cloud-API)
conda activate smol
python zsibot_vln/llms/smolvlm2_256m_video_instruct/server.py
```

```bash
#run MATRiX on shell 1 (no conda)
cd matrix && export ROS_DOMAIN_ID=0 && source /opt/ros/humble/setup.bash && ./run_sim.sh 1 6
# ==== NOTE ====
# Remember to stand the robot up using LB+Y (controller mode) or "u" (keyboard control mode).
```

```bash
#run env_bridge on shell 2 (no conda)
cd zsibot_vln/bridge && export ROS_DOMAIN_ID=0 && source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash && ros2 run env_bridge env_bridge
```

```bash
#run mc_sdk_bridge on shell 3 (no conda)
cd zsibot_vln/bridge && export ROS_DOMAIN_ID=0 && source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash && ros2 run mc_sdk_bridge mc_sdk_bridge
```

```bash
#run the baselie model on shell 4
cd zsibot_vln && conda activate zsibot_vln
#search using an open-vocabulary text goal
python main.py --goal_type text --text_goal "green plant"
#or
#search using an image goal
python main.py --goal_type ins_image --image_goal_path ./goals/bed.jpg
```




## 🤝 Acknowledgments

This project builds upon and acknowledges the following works:

[MATRiX](https://github.com/zsibot/matrix) – a robotic simulation framework featuring realistic scene rendering and physical dynamics.

[UniGoal](https://github.com/bagh2178/UniGoal) – a zero-shot VLN method leveraging LLMs.

## 📄 License
This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.
