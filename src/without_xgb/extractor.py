import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import psutil, time
import matplotlib.pyplot as plt
import torchvision

try:
    import timm  # for EfficientNet, ConvNeXt, etc.
except ImportError:
    timm = None
    print("⚠️ timm not installed. Run: pip install timm open_clip_torch")


class PolicyNet(nn.Module):
    def __init__(self, num_actions, backbone="resnet50", device=None, image_dir="dataset/images/"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        self.backbone_name = backbone.lower()

        # 1️⃣ Backbone selection
        self.backbone, feat_dim, preprocess = self._load_backbone(self.backbone_name)

        # 2️⃣ Policy head
        self.policy = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # 3️⃣ Image preprocessing
        self.transform = preprocess or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.to(self.device)

    # --------------------------------------------
    # Load backbone network
    # --------------------------------------------
    def _load_backbone(self, name):
        if name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048

        elif name == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            backbone = nn.Sequential(base.features, base.avgpool)
            feat_dim = 960

        elif name == "efficientnet_b0":
            if timm:
                backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
                feat_dim = backbone.num_features
            else:
                raise ImportError("Install timm for EfficientNet support.")

        elif name == "clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            backbone = model.visual
            feat_dim = model.visual.output_dim
            return backbone, feat_dim, preprocess

        elif name == "dinov2":
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            feat_dim = 768
            preprocess = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            return backbone, feat_dim, preprocess

        else:
            raise ValueError(f"Unknown backbone: {name}")

        return backbone, feat_dim, None

    # --------------------------------------------
    # Forward pass
    # --------------------------------------------
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = torch.flatten(feats, 1)
        logits = self.policy(feats)
        return logits

    # --------------------------------------------
    # Inference (single image)
    # --------------------------------------------
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
        return action

    # --------------------------------------------
    # Supervised behavior cloning training
    # --------------------------------------------
    def fit_supervised(self, dataset_df, num_epochs=10, lr=1e-4):
        """
        Behavior Cloning training with runtime and memory logging.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss  # more accurate than virtual_memory()

        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc=f"Epoch {epoch+1}"):
                path = os.path.join(self.image_dir, row['file'])
                label = torch.tensor([row['label']], dtype=torch.long).to(self.device)
                img = Image.open(path).convert('RGB')
                tensor = self.transform(img).unsqueeze(0).to(self.device)

                logits = self.forward(tensor)
                loss = criterion(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Loss: {total_loss / len(dataset_df):.4f}")

        end_time = time.time()
        end_mem = psutil.Process(os.getpid()).memory_info().rss

        # Metrics
        training_time_sec = end_time - start_time
        memory_used_MB = (end_mem - start_mem) / (1024 ** 2)
        model_size_MB = os.path.getsize(f"bc_{self.backbone_name}.pth") / (1024 ** 2) if os.path.exists(f"bc_{self.backbone_name}.pth") else 0

        print(f"🕒 Training time: {training_time_sec / 60:.2f} min")
        print(f"💾 Memory used (RAM): {memory_used_MB:.2f} MB")
        print(f"📦 Model size: {model_size_MB:.2f} MB")

        # Return stats for metrics.json
        return {
            "training_time_sec": round(training_time_sec, 2),
            "training_time_min": round(training_time_sec / 60, 2),
            "memory_used_MB": round(memory_used_MB, 2),
            "model_size_MB": round(model_size_MB, 2)
        }
