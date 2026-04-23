AdaSAM-AD
AdaSAM-AD is a deep learning framework designed for medical image segmentation and anomaly detection. It leverages the power of Segment Anything Model 2 (SAM-2) with adaptive modules to achieve state-of-the-art performance in complex medical imaging tasks, such as colon polyp detection.

🛠 Prerequisites
Environment Setup
The project requires Python 3.10+ and CUDA 12.1 for optimal performance.

Core Dependencies
Ensure the following libraries are installed in your environment:

SAM-2 Installation
This project relies on a local installation of the SAM-2 repository.

Clone the official repository: https://github.com/facebookresearch/sam2

Follow the official instructions to install it locally (e.g., pip install -e .).

🚀 Getting Started
1. Installation
First, clone this repository and install the specific versions of the required packages:

2. Data Preparation
Place your medical datasets (e.g., Kvasir-SEG, CVC-ClinicDB) in the data/ directory. Ensure the structure follows the expected format:

data/train/images

data/train/masks

3. Training
To start the training process using the default configuration, run:

📈 Key Features
SAM-2 Integration: Seamlessly utilizes the hierarchical features and temporal memory of Segment Anything Model 2.

Adaptive Architecture: Specifically optimized for medical imaging nuances like low contrast and varying object scales.

Modular Design: Easy-to-extend components for research on different medical segmentation targets.

📝 Citation
If you find this work useful for your research, please cite our project:
