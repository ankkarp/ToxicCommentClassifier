import numpy as np


def f1_score(y_pred, y_true):
    """
    Подсчитывает f1 score для двух целевых классов

    Параметры:
        y_pred: массив из булевых/численных значений, например из true/false или 0/1
            массив пресказанных классов
        y_true: массив из булевых/численных значений, например из true/false или 0/1
            массив истинных классов
    """
    y_pred = np.array(y_pred, dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    tp = (y_pred & y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fn = (y_pred & ~y_true).sum()
    fp = (~y_pred & y_true).sum()
    if tp == 0 or tp + fp == 0 or tp + fn == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
