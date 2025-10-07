import os, json, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image
from datetime import datetime

class MetricsEvaluator:
    def __init__(self, model, metrics_dir="metrics"):
        self.model = model
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir = metrics_dir

    def evaluate(self, splits):
        self.model.eval()
        results = {}
        backbone = getattr(self.model, "backbone_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with torch.no_grad():
            for split_name in ["train", "val", "test"]:
                y_true, y_pred = [], []
                df = splits[split_name]

                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {split_name}"):
                    path = os.path.join(self.model.image_dir, row["file"])
                    label = int(row["label"])

                    img = Image.open(path).convert("RGB")
                    tensor = self.model.transform(img).unsqueeze(0).to(self.model.device)

                    logits = self.model.forward(tensor)
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()

                    y_true.append(label)
                    y_pred.append(pred)

                results[split_name] = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                    "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                }

        metrics_path = os.path.join(self.metrics_dir, f"metrics_{backbone}_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)

        model_path = f"bc_{backbone}_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_path)

        print(f"✅ Evaluation complete!")
        print(f"📊 Metrics saved to {metrics_path}")
        print(f"💾 Model saved as {model_path}")
        return results
