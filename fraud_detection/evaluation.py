"""Metrics distinguish thresholded predictions from continuous scores."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(y_true, scores, predictions):
    """Return held-out metrics and confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {'precision': float(precision_score(y_true, predictions, zero_division=0)),
            'recall': float(recall_score(y_true, predictions, zero_division=0)),
            'f1': float(f1_score(y_true, predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, scores)),
            'average_precision': float(average_precision_score(y_true, scores)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'predicted_anomalies': int(np.sum(predictions)),
            'predicted_anomaly_percentage': float(np.mean(predictions) * 100)}
