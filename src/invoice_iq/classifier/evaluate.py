"""Classification metrics (accuracy, macro-F1, confusion matrix) in pure numpy.

Computed directly rather than via scikit-learn to keep dependencies light and to
make the math explicit. These feed `models/metrics.json`, which is auto-synced
into the README.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class ClassificationMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    n_test: int


def confusion_matrix(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], num_classes: int
) -> NDArray[np.int64]:
    """Rows = actual class, columns = predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[int(true), int(pred)] += 1
    return matrix


def _per_class_f1(matrix: NDArray[np.int64]) -> NDArray[np.float64]:
    f1s = np.zeros(matrix.shape[0], dtype=np.float64)
    for c in range(matrix.shape[0]):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp) / denom if denom > 0 else 0.0
    return f1s


def compute_metrics(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], num_classes: int
) -> ClassificationMetrics:
    """Compute accuracy, macro-F1, per-class F1, and the confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    per_class = _per_class_f1(matrix)
    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        macro_f1=round(float(per_class.mean()), 4),
        per_class_f1=[round(float(x), 4) for x in per_class],
        confusion_matrix=matrix.tolist(),
        n_test=int(len(y_true)),
    )
