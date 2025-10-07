# 🛴 Ecomobility — Behavior Cloning Model

This repository implements a **Behavior Cloning** framework for Reinforcement Learning–based autonomous e-scooter navigation in smart city environments.
The codebase integrates **deep visual feature extraction (ResNet-50 or Foundation Models)** with XGBoost or neural policy networks for decision-making, enabling imitation-based learning from human driving demonstrations.

---

## ⚙️ System Specifications

| Component | Specification |
|------------|---------------|
| **CPU** | Intel Core i7-12700KF (12 cores / 20 threads, 3.6–5.0 GHz) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM) |
| **CUDA Version** | 12.9 |
| **Python Version** | 3.11 |

---

## 🧠 Project Overview

The main workflow includes:

1. **Dataset Preparation** — preprocessing and labeling of e-scooter trajectory images.
2. **Feature Extraction** — embedding generation using pretrained CNN backbones (e.g. ResNet-50).
3. **Policy Learning** — supervised imitation learning via either:
   - Neural network (`PolicyNet`) with focal loss, or  
   - XGBoost classifier for high-level decision policies.
4. **Evaluation** — automated metrics computation (accuracy, F1-score, confusion matrix).

---

## 🚀 Installation and Setup

Clone the repository and prepare the environment:

```bash
# 1️⃣ Clone repository
git clone https://github.com/icsa-hua/Ecomobility---Behavior_Cloning_Model.git
cd Ecomobility---Behavior_Cloning_Model

# 2️⃣ Create virtual environment
python -m venv ecomobility_env
# Windows
ecomobility_env\Scripts\activate
# macOS/Linux
source ecomobility_env/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

## 🧩 Running Experiments

### 🔹 Neural Policy (PyTorch-based)

To train and evaluate the neural imitation policy, navigate to the **`src/without_xgb/`** directory and specify the desired backbone:

```bash
cd src/without_xgb

# Example runs
python main.py --backbone resnet50
python main.py --backbone efficient_b0
python main.py --backbone clip
python main.py --backbone mobilenet_v3_large
python main.py --backbone dinov
```
Each run will automatically:

- Select the appropriate model backbone  
- Train (`PolicyNet`) using supervised learning  
- Save weights and metrics with timestamps (`bc_<backbone>_<timestamp>.pth`)  

### 🔹 XGBoost Policy

To train a lightweight policy using XGBoost on top of precomputed visual features, go to the **`src/xgb/`** directory and execute:

```bash
cd src/xgb
python main.py
```


This pipeline:

- Loads visual embeddings extracted by pretrained CNNs  
- Trains an XGBoost classifier on action labels  
- Evaluates model performance using the same unified metrics module  
