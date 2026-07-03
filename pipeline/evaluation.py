"""
Model evaluation helpers.

Provides a single public ``test_model()`` dispatcher and the underlying
``test_nn_model()`` / ``test_sklearn_model()`` implementations, plus the
``compute_metrics()`` utility for deriving accuracy, recall and F1 from a
confusion matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from sklearn.metrics import confusion_matrix
from typing import Tuple

from pbi_models.classifiers.sklearn_classifier import SklearnClassifier
from pbi_utils.logging import Logging
from pipeline.data import dataframe_to_tf_dataloader, dataframe_to_numpy_X_y

logger = Logging()


def compute_metrics(
    tn: float, fp: float, fn: float, tp: float
) -> Tuple[float, float, float]:
    """
    Compute Accuracy, Recall and F1Score from the confusion matrix.

    :param tn: True Negatives.
    :param fp: False Positives.
    :param fn: False Negatives.
    :param tp: True Positives.
    :return: ``(accuracy, recall, f1)``.
    """
    acc = (tp + tn) / (tp + tn + fp + fn)
    rec = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return acc, rec, f1


def test_model(
    test_df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Test a model (PyTorch nn.Module or SklearnClassifier) on *test_df*.

    :param test_df: DataFrame with columns ``bacterium_embedding``,
        ``phage_embedding``, and ``interaction_type``.
    :param model: Trained model to evaluate.
    :param batch_size: Batch size (used for NN models only).
    :param device: Device string (e.g. ``"cpu"`` or ``"cuda:0"``).
    :param silent: Suppress log output.
    :return: ``(confusion_matrix, test_loss)``.  ``test_loss`` is ``-1`` for sklearn
        models.  Confusion matrix format: ``[[tn, fp], [fn, tp]]``.
    """
    if isinstance(model, nn.Module):
        return test_nn_model(test_df, model, batch_size, device, silent)
    else:
        return test_sklearn_model(test_df, model, batch_size, device, silent)


def test_nn_model(
    test_df: pd.DataFrame,
    model: nn.Module,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Evaluate a PyTorch nn.Module on *test_df*.

    See :func:`test_model` for parameter descriptions.
    """
    if not silent:
        logger.info("Starting testing...")

    dataloader = dataframe_to_tf_dataloader(test_df, batch_size, device)
    criterion = nn.CrossEntropyLoss()

    # Metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm_metric = tm.ConfusionMatrix(task="binary").to(device)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for bact_emb, phg_emb, labels in dataloader:
            logits = model(bact_emb, phg_emb)
            loss = criterion(logits, labels)
            test_loss += loss.item() * bact_emb.size(0)

            predictions = logits.argmax(dim=1, keepdim=True).squeeze()

            if not silent:
                accuracy(predictions, labels)
                f1(predictions, labels)
                recall(predictions, labels)
            cm_metric(predictions, labels)

    cm_mat = cm_metric.compute().cpu().numpy()  # torchmetrics default: TN, FP, FN, TP
    test_loss = test_loss / len(dataloader.dataset)  # type: ignore
    tn, fp, fn, tp = cm_mat.ravel().tolist()

    if not silent:
        acc, rec, f1_val = compute_metrics(tn, fp, fn, tp)
        logger.info(f"Accuracy (test): {accuracy.compute()}")
        logger.info(f"Recall (test): {recall.compute()}")
        logger.info(f"F1 score (test): {f1.compute()}")
        logger.info(f"Loss (test): {test_loss}")
        logger.info(
            f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}"
        )

    return cm_mat, test_loss


def test_sklearn_model(
    test_df: pd.DataFrame,
    model: SklearnClassifier,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Evaluate a SklearnClassifier on *test_df*.

    See :func:`test_model` for parameter descriptions.
    """
    if not silent:
        logger.info("Starting testing...")

    X_test, y_test = dataframe_to_numpy_X_y(test_df)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    if not silent:
        logger.info(f"Accuracy (test): {acc}")
        logger.info(f"Recall (test): {rec}")
        logger.info(f"F1 score (test): {f1}")
        logger.info(
            f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}"
        )

    return cm, -1
