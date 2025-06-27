
import numpy as np
from sklearn import metrics as skmetrics
from sklearn.metrics import precision_score, hamming_loss


def AVGF1(y_pre, y_true):
    total = 0
    p_total = 0
    p, r = 0, 0
    for yt, yp in zip(y_true, y_pre):
        ytNum = sum(yt)
        if ytNum == 0:
            continue
        rec = sum(yp[yt == 1]) / ytNum
        r += rec
        total += 1
        ypSum = sum(yp)
        if ypSum > 0:
            p_total += 1
            pre = sum(yt[yp == True]) / ypSum
            p += pre
    r /= total
    if p_total > 0:
        p /= p_total
    return 2 * r * p / (r + p)


def PrecisionInTop(Y_prob_pre, Y, n):
    Y_pre = np.argsort(1 - Y_prob_pre, axis=1)[:, :n]
    return sum([sum(y[yp]) for yp, y in zip(Y_pre, Y)]) / (len(Y) * n)


def evaluate_loc(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    mip = skmetrics.precision_score(y_true, y_pred, average='micro')
    mir = skmetrics.recall_score(y_true, y_pred, average='micro')
    mif = skmetrics.f1_score(y_true, y_pred, average='micro')
    miauc = skmetrics.roc_auc_score(y_true, y_prob, average='micro')
    maauc = skmetrics.roc_auc_score(y_true, y_prob, average='macro')
    hl = hamming_loss(y_true, y_pred)
    ap = precision_score(y_true, y_pred, average='samples')
    avgf1 = AVGF1(y_pred, y_true)
    pat1 = PrecisionInTop(y_prob, y_true, n=1)
    return mip, mir, mif, miauc, maauc, hl, ap, avgf1, pat1
