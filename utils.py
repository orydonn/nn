import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def weighted_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    weights_dict: dict,
    label_encoder: LabelEncoder,
) -> float:
    """Compute weighted one-vs-all ROC-AUC."""
    present_labels = np.unique(y_true)
    if present_labels.size == 0:
        return 0.0
    try:
        aucs = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class="ovr",
            average=None,
            labels=present_labels,
        )
    except ValueError:
        return 0.0
    labels_str = label_encoder.inverse_transform(present_labels)
    weights = np.array([weights_dict.get(str(lbl), 0) for lbl in labels_str], dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights.sum()
    min_len = min(len(aucs), len(weights))
    return float(np.sum(aucs[:min_len] * weights[:min_len]))
