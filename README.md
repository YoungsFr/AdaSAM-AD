# AdaSAM-AD

This repository is the official implementation of AdaSAM-AD: Boosting SAM2 for Fine-grained Pixel-level Anomaly Detection via Spatial-Channel Calibration and Deformable Cascades

## Visualization
<div align="center">
  <!-- SAVE FIGURE 1 or 6 FROM THE PDF AS assets/visuals.png -->
  <img src="heatmap.png" width="90%">
  <br>
  <em>Visual comparison between ESCNet and other SOTA methods. Our model accurately segments objects with complex backgrounds and intricate boundaries.</em>
</div>


## 🔧 Requirements

Please install the following dependencies:

torch==2.5.1+cu121

torchvision==0.20.1+cu121

einops==0.8.1

opencv-python==4.13.0.90

numpy==2.3.4

pillow==10.2.0

matplotlib==3.10.8

## SAM-2 Installation

Clone and install SAM-2 manually:

git clone https://github.com/facebookresearch/sam2

cd sam2

pip install -e .

## Run

python train.py

