import os, time, json, psutil, torch, argparse
from datetime import datetime
from prepare import DatasetPrep
from extractor import PolicyNet
from metrics import MetricsEvaluator
from PIL import Image
from time import perf_counter
import random
import numpy as np

# -----------------------------
# 1️⃣ Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Behavior Cloning Training Script")
parser.add_argument(
    "--backbone",
    type=str,
    default="resnet50",
    choices=["resnet50", "efficient_b0", "clip", "mobilenet_v3_large", "dinov2"],
    help="Select backbone architecture"
)
args = parser.parse_args()

backbone_name = args.backbone
print(f"🧠 Selected backbone: {backbone_name}")

# -----------------------------
# 2️⃣ Reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"✅ Random seed fixed to {seed} for reproducibility")

# -----------------------------
# 3️⃣ Dataset loading
# -----------------------------
prep = DatasetPrep("../dataset/art_labels.csv")
splits = prep.prepare()

# -----------------------------
# 4️⃣ Device selection
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple MPS GPU backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("⚡ Using NVIDIA CUDA GPU backend")
else:
    device = torch.device("cpu")
    print("💻 Using CPU backend")

# -----------------------------
# 5️⃣ Model setup
# -----------------------------
num_actions = len(splits['train']['label'].unique())
model = PolicyNet(num_actions, backbone=backbone_name, device=device)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"bc_{backbone_name}_{timestamp}.pth"
metrics_path = f"metrics_{backbone_name}_{timestamp}.json"

metrics_extra = {}

# -----------------------------
# 6️⃣ Train or load pretrained model
# -----------------------------
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded pretrained model ({model_path})")
else:
    print(f"⚙️ Training new {backbone_name} policy...")
    profiling_stats = model.fit_supervised(splits['train'], num_epochs=10, lr=1e-4)
    metrics_extra.update(profiling_stats)

    torch.save(model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / (1024 ** 2)
    metrics_extra["model_size_MB"] = round(model_size, 2)
    print(f"✅ Training completed. Model saved at {model_path}")

# -----------------------------
# 7️⃣ Evaluation (accuracy, F1, etc.)
# -----------------------------
evaluator = MetricsEvaluator(model)
metrics = evaluator.evaluate(splits)

# -----------------------------
# 8️⃣ Inference benchmark
# -----------------------------
test_imgs = splits["test"]["file"].sample(min(50, len(splits["test"])))
inference_times = []
process = psutil.Process(os.getpid())

for img_name in test_imgs:
    img_path = os.path.join(model.image_dir, img_name)
    tensor = model.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(model.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = perf_counter()
    _ = model.forward(tensor)
    end = perf_counter()
    inference_times.append(end - start)

avg_inf = sum(inference_times) / len(inference_times)
metrics_extra["avg_inference_time_sec"] = round(avg_inf, 5)
metrics_extra["avg_inference_fps"] = round(1.0 / avg_inf, 2)
metrics_extra["current_ram_MB"] = round(process.memory_info().rss / (1024 ** 2), 2)
if torch.cuda.is_available():
    metrics_extra["gpu_memory_MB"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

# -----------------------------
# 9️⃣ Save all results
# -----------------------------
all_metrics = {
    "backbone": backbone_name,
    "device": str(device),
    "timestamp": timestamp,
    "profiling": metrics_extra,
    "results": metrics
}

with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"📊 Evaluation complete! Metrics saved to {metrics_path}")
