from prepare import DatasetPrep
from extractor import ResNet50XGBClassifier
from metrics import MetricsEvaluator

prep = DatasetPrep("dataset/art_labels.csv")
splits = prep.prepare()

extractor = ResNet50XGBClassifier("dataset/images/")
clf = extractor.fit(splits)

evaluator = MetricsEvaluator(clf)
metrics = evaluator.evaluate(clf, splits, extractor)

print("Evaluation complete! Metrics saved to metrics.json, model saved to model.xgb")
