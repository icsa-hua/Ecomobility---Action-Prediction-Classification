import os, json, joblib, time, psutil
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class MetricsEvaluator:
    def __init__(self, clf, save_path="model.xgb", metrics_dir="metrics"):
        self.clf = clf
        self.save_path = save_path
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir = metrics_dir

    def evaluate(self, clf, splits, extractor):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backbone = getattr(extractor, "backbone_name", "resnet50_xgb")

        # --- Profiling ---
        start = time.time()
        process = psutil.Process(os.getpid())

        results = {}
        for split_name in ["train", "val", "test"]:
            X = extractor.extract_embeddings(splits[split_name])
            y_true = splits[split_name]['label'].values
            y_pred = clf.predict(X)

            results[split_name] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }

        end = time.time()

        profiling = {
            "training_time_sec": round(end - start, 2),
            "training_time_min": round((end - start) / 60, 2),
            "current_ram_MB": round(process.memory_info().rss / (1024 ** 2), 2),
            "model_size_MB": round(os.path.getsize(self.save_path) / (1024 ** 2), 2)
        }

        # --- JSON ---
        all_metrics = {
            "backbone": backbone,
            "timestamp": timestamp,
            "profiling": profiling,
            "results": results
        }

        # Save everything
        metrics_path = os.path.join(self.metrics_dir, f"metrics_{backbone}_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        joblib.dump(self.clf, self.save_path)

        print(f"✅ Evaluation complete!")
        print(f"📊 Metrics saved to {metrics_path}")
        print(f"💾 Model saved as {self.save_path}")

        return all_metrics
