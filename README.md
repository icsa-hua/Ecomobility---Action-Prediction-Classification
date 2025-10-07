# 🛴 Ecomobility — Behavior Cloning Model

This repository implements a **Behavior Cloning** framework for Reinforcement Learning–based autonomous e-scooter navigation in smart city environments.
The codebase integrates **deep visual feature extraction (ResNet-50 or Foundation Models)** with XGBoost or neural policy networks for decision-making, enabling imitation-based learning from human driving demonstrations.

---

## ⚙️ System Specifications

| Component | Specification |
|------------|---------------|
| **Operating System** | Windows 11 Pro 64-bit |
| **CPU** | Intel Core i7-12700KF (12 cores / 20 threads, 3.6–5.0 GHz) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM) |
| **CUDA Version** | 12.9 |
| **Python Version** | 3.11 |
| **PyTorch Version** | 2.2+ (with CUDA backend) |

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
git clone https://github.com/<username>/Ecomobility---Behavior-Cloning-Model.git
cd Ecomobility---Behavior-Cloning-Model

# 2️⃣ Create virtual environment
python -m venv ecomobility_env
# Windows
ecomobility_env\Scripts\activate
# macOS/Linux
source ecomobility_env/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

