import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class MetricsEvaluator:
    def __init__(self, clf, save_path="model.xgb", metrics_file="metrics.json"):
        self.clf = clf
        self.save_path = save_path
        self.metrics_file = metrics_file

    def evaluate(self, clf, splits, extractor):
        results = {}
        for split_name in ["train", "val", "test"]:
            X = extractor.extract_embeddings(splits[split_name])
            y_true = splits[split_name]['label'].values
            y_pred = clf.predict(X)

            results[split_name] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted"),
                "report": classification_report(y_true, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }

        # Save metrics
        with open(self.metrics_file, "w") as f:
            json.dump(results, f, indent=4)

        # Save model
        joblib.dump(self.clf, self.save_path)

        return results
