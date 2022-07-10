import numpy as np


def f1_score(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    tp = (y_pred & y_true).sum()
    tn = (np.logical_not(y_pred) & np.logical_not(y_true)).sum()
    fn = (y_pred & np.logical_not(y_true)).sum()
    fp = (np.logical_not(y_pred) & y_true).sum()
    if tp+fp == 0 or tp+fn == 0:
        return (y_pred == y_true).sum()
    print(tp, tn, fn, fp)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    for i in range(100):
        a = np.random.randint(low=0, high=2, size=16)
        b = np.random.randint(low=0, high=2, size=16)
        print(a, b)
        print(f1_score(a, b))
        print((a == b).sum() / len(a))

