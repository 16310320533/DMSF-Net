# coding=utf8
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


def calculate_metrics(y_true, y_pred):
    """
    Computes segmentation metrics: F1, IoU, Precision, Recall, Accuracy.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = (y_pred == y_true).mean()
    dice = 2 * (precision * recall) / (precision + recall + 1e-6)

    return accuracy, f1, iou, precision, recall, dice


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False