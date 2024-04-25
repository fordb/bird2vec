import numpy as np
from evaluate import load


# define eval metrics
accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")
confusion_matrix = load("confusion_matrix")
# TODO: fix this
# roc_auc_metric = load("roc_auc", "multiclass")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    prec = precision_metric.compute(predictions=preds, references=labels, average="macro")
    rec = recall_metric.compute(predictions=preds, references=labels, average="macro")
    f1_score = f1_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_score["f1"],
        # "roc_auc": roc_auc["roc_auc"],
    }