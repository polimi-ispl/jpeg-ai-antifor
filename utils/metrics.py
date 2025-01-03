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
def compute_all_metrics(scores_df_1: pd.DataFrame, scores_df_2: pd.DataFrame) -> List[float]:
    """
    Simple function to compute all the metrics on the scores distribution of the detectors considered in the paper.
    In particular, we compute:
    - Earth Mover's Distance (Wasserstein Distance)
    - Area Under the ROC Curve
    - Balanced Accuracy at threshold = 0
    - True Positive Rate at threshold = 0
    - False Positive Rate at threshold = 0
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

    # BALANCED ACC at thr=0
    ba_thr0 = (1 - fpr_thr0 + tpr_thr0) / 2

    return wd_score, auc, fpr_thr0, tpr_thr0, ba_thr0