import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import xgboost as xgb
import numpy as np
from tqdm import tqdm

class ResNet50XGBClassifier:
    def __init__(self, image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_dir = image_dir
        self.device = device
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        self.clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    def extract_embeddings(self, df):
        embeddings = []
        for f in tqdm(df['file'], desc="Extracting embeddings"):
            path = os.path.join(self.image_dir, f)
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(tensor).squeeze().cpu().numpy()
            embeddings.append(emb)
        return np.vstack(embeddings)

    def fit(self, splits):
        X_train = self.extract_embeddings(splits['train'])
        y_train = splits['train']['label'].values

        self.clf.fit(X_train, y_train)
        return self.clf
