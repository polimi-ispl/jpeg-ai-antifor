"""
Simple file to compute metrics on the scores distribution of the detectors.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
from typing import List

# --- Functions --- #

"""
Metrics for the deepfake image detectors
"""

def compute_all_metrics(scores_df_1: pd.DataFrame, scores_df_2: pd.DataFrame) -> List[float]:
    """
    Simple function to compute all the metrics on the scores distribution of the detectors considered in the paper.
    In particular, we compute:
    - Earth Mover's Distance (Wasserstein Distance)
    - Area Under the ROC Curve
    - Balanced Accuracy at threshold = 0
    - True Positive Rate at threshold = 0
    - False Positive Rate at threshold = 0
    - False Negative Rate at threshold = 0
    - True Negative Rate at threshold = 0
    We consider conventionally the first Dataframe as containing negative values, the second positive values

    :param scores_df_1: the first DataFrame containing the scores
    :param scores_df_2: the second DataFrame containing the scores
    :return: the computed metrics
    """

    # --- Distribution metrics

    # WD
    wd_score = wasserstein_distance(scores_df_1['logits'], scores_df_2['logits'])

    # --- Detection metrics

    # Create the labels
    labels = np.concatenate([np.zeros_like(scores_df_1['logits']), np.ones_like(scores_df_2['logits'])])
    scores = np.concatenate([scores_df_1['logits'].tolist(), scores_df_2['logits'].tolist()])

    # compute roc curve parameters
    auc = roc_auc_score(labels, scores)

    # FPR at thr=0
    negative_values = scores_df_1['logits'].values
    fpr_thr0 = len(negative_values[negative_values > 0]) / len(negative_values)

    # TPR at thr=0
    positive_values = scores_df_2['logits'].values
    tpr_thr0 = len(positive_values[positive_values > 0]) / len(positive_values)

    # FNR at thr=0
    fnr_thr0 = len(positive_values[positive_values <= 0]) / len(positive_values)

    # TNR at thr=0
    tnr_thr0 = len(negative_values[negative_values <= 0]) / len(negative_values)

    # BALANCED ACC at thr=0
    ba_thr0 = (tnr_thr0 + tpr_thr0) / 2

    return wd_score, auc, fpr_thr0, tpr_thr0, ba_thr0, fnr_thr0, tnr_thr0

"""
Splicing localization metrics from TruFor repository (https://github.com/grip-unina/TruFor)
"""

def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)

    inds = np.argsort(values)
    inds = inds[(gt0[inds] + gt1[inds]) > 0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]

    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN

    msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]

    return FP, TP, FN, TN, vet_th


def computeMetrics_th(values, gt, gt0, gt1, th):
    values = values > th
    values = values.flatten().astype(np.uint8)
    gt = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)

    gt = gt[(gt0 + gt1) > 0]
    values = values[(gt0 + gt1) > 0]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, values, labels=[0, 1])

    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TP = cm[1, 1]

    return FP, TP, FN, TN


def computeMCC(FP, TP, FN, TN):
    FP = np.float64(FP)
    TP = np.float64(TP)
    FN = np.float64(FN)
    TN = np.float64(TN)
    return np.abs(TP * TN - FP * FN) / np.maximum(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-32)


def computeF1(FP, TP, FN, TN):
    return 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)


def calcolaMetriche_threshold(mapp, gt, th):
    FP, TP, FN, TN = computeMetrics_th(mapp, gt, th)

    f1 = computeF1(FP, TP, FN, TN)
    f1i = computeF1(TN, FN, TP, FP)
    maxF1 = max(f1, f1i)

    return 0, maxF1, 0, 0, 0


def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)

    # best threshold
    try:
        FP, TP, FN, TN, _ = computeMetricsContinue(map, gt0, gt1)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan

    # fixed threshold
    try:
        FP, TP, FN, TN = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
        FPR_05= FP / (FP + TN) if (FP + TN) > 0 else np.nan
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan
        FPR_05 = np.nan

    return F1_best, F1_th, FPR_05


def computeDetectionMetrics(scores, labels):
    lbl = np.array(labels)
    lbl = lbl[np.isfinite(scores)]

    scores = np.array(scores, dtype='float32')
    scores[scores == np.PINF] = np.nanmax(scores[scores < np.PINF])
    scores = scores[np.isfinite(scores)]
    assert lbl.shape == scores.shape

    # AUC
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    AUC = roc_auc_score(lbl, scores)

    # Balanced Accuracy
    from sklearn.metrics import balanced_accuracy_score
    fpr, tpr, _ = roc_curve(lbl, scores)
    tnr = 1 - fpr  # True Negative Rate
    ba = (tpr + tnr) / 2  # Balanced Accuracy
    bACC_best = ba.max()

    bACC = balanced_accuracy_score(lbl, scores > 0.5)

    return AUC, bACC, bACC_best