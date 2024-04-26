import wandb
import numpy as np
from evaluate import load
from sklearn.metrics import confusion_matrix


# define eval metrics
accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")
# TODO: fix this
# roc_auc_metric = load("roc_auc", "multiclass")

def compute_metrics(pred, class_names):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    prec = precision_metric.compute(predictions=preds, references=labels, average="macro", zero_division=0)
    rec = recall_metric.compute(predictions=preds, references=labels, average="macro", zero_division=0)
    f1_score = f1_metric.compute(predictions=preds, references=labels, average="macro")

    # log confusion matrix
    cf_matrix = wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels,
        preds=preds,
        class_names=class_names
    )
    wandb.log({"confusion_matrix": cf_matrix})

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_score["f1"],
        # "roc_auc": roc_auc["roc_auc"],
    }