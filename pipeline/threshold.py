"""
Provides :func:`sweep_thresholds` to find the decision threshold that
maximises F1 on a labelled probability array.  This is used internally by
the pipeline when ``calibrate_threshold=True`` is set in the configuration.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximises F1 score.

    Iterates over *thresholds*, computes F1 at each, and returns the threshold
    with the highest F1.

    :param probs: 1-D array of predicted positive-class probabilities (values in
        ``[0, 1]``), e.g. ensemble-averaged ``softmax[:, 1]``.
    :param labels: 1-D integer array of ground-truth labels (``0`` or ``1``).
    :param thresholds: Thresholds to sweep.  Defaults to
        ``np.arange(0.01, 1.0, 0.01)`` (99 values).
    :return: ``(best_threshold, best_f1)`` where *best_threshold* is the
        threshold that produced the highest F1 on *probs* / *labels*.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = 0.0
    best_thr = 0.5  # sensible default if no threshold beats 0

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1
