import numpy as np
from datasets import load_metric


# define eval metrics
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")


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
        "f1": f1_score["f1"]
    }