### Installation of baseline VLN Model

A gentle reminder: connection errors may occur, please wait or try again. VPN is recommended.

```bash
cd zsibot_vln
conda create --name zsibot_vln python=3.8 -y
conda activate zsibot_vln

# LightGlue
pip install git+https://github.com/cvg/LightGlue.git 
# Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git # A stable VPN is recommended

# Detectorn weights 
URL="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
DEST="~/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600"
mkdir -p $DEST
wget -O $DEST/model_final_f10217.pkl $URL

# PyTorch3D (building wheels may take a long time)
pip install git+https://github.com/facebookresearch/pytorch3d.git
```

```bash
# Grounded-Segment-Anything
cd zsibot_vln/agents/zeroshot/unigoal
mkdir -p third_party
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git third_party/Grounded-Segment-Anything #vpn is recommended here
cd third_party/Grounded-Segment-Anything
git checkout 5cb813f
# Editable installs
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO # turn off vpn for this
# Back to repo root
cd ../../
```

```bash
# Model weights
mkdir -p third_party/weights/
wget -O third_party/weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O third_party/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

```bash
# Project dependencies
conda install pytorch::faiss-gpu -y
# Python deps
pip install -r requirements.txt
```

