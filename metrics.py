import torch
import numpy as np
from lifelines.utils import concordance_index

def get_all_metrics(outputs, y, threshold):
    score = score_metric(outputs, y, threshold)
    precision = precision_metric(outputs, y, threshold)
    recall = recall_metric(outputs, y, threshold)
    f1 = f1_score_metric(outputs, y, threshold)
    above_th_predictions = above_threshold_prediction_number(outputs, threshold)
    return {"score":score, "precision":precision, "recall":recall, "f1":f1, "above_threshold_predictions":above_th_predictions}

def general_metrics(y_pred, y_true, threshold):
    preds = y_pred.detach().clone()
    preds[preds > threshold] = 1.
    preds[preds <= threshold] = 0.
    correct = (preds==y_true).sum().item()
    wrong = (preds!=y_true).sum().item()
    tp = (preds*y_true).sum().item()
    tn = correct - tp
    fp = (torch.logical_not(y_true)*preds).sum().item()
    fn = wrong - fp
    return tp, tn, fp, fn

def precision_metric(y_pred, y_true, threshold):
    tp, tn, fp, fn = general_metrics(y_pred, y_true, threshold)
    return tp / (tp + fp + 1e-9)

def recall_metric(y_pred, y_true, threshold):
    tp, tn, fp, fn = general_metrics(y_pred, y_true, threshold)
    return tp / (tp + fn + 1e-9)

def f1_score_metric(y_pred, y_true, threshold):
    p = precision_metric(y_pred, y_true, threshold)
    r = recall_metric(y_pred, y_true, threshold)
    return 2*p*r/(p+r + 1e-9)

def score_metric(y_pred, y_true, threshold):
    """
    This metric is monitored to determine the best model.
    """
    return f1_score_metric(y_pred, y_true, threshold) + precision_metric(y_pred, y_true, threshold)

def above_threshold_prediction_number(y_pred, threshold):
    return (y_pred>threshold).sum()

def ci_index(y_pred,y_true):
    return concordance_index(y_true, y_pred)

if __name__ == "__main__":
    y_pred = np.array([0.7,0.2,0.6])
    y_true = np.array([1.,1.,0.])
    a = score(torch.from_numpy(y_pred), torch.from_numpy(y_true), 0.5)
    print(a)
