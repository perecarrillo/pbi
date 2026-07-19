"""
Model evaluation helpers.

Provides a single public ``test_model()`` dispatcher and the underlying
``test_nn_model()`` / ``test_sklearn_model()`` implementations, plus the
``compute_metrics()`` utility for deriving accuracy, recall and F1 from a
confusion matrix.
``test_nn_ensemble()``: evaluate a list of models as an ensemble (averaged
  softmax) with a configurable decision threshold.
``get_ensemble_probabilities()``: return raw ensemble-averaged probabilities
  (needed by the threshold calibration sweep).
``test_model()`` dispatcher updated to accept a list of models and an optional
  threshold parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from sklearn.metrics import confusion_matrix
from typing import List, Tuple

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
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return acc, rec, f1


def test_model(
    test_df: pd.DataFrame,
    model: nn.Module | List[nn.Module] | SklearnClassifier,
    batch_size: int,
    device: str,
    threshold: float = 0.5,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Test a model (or ensemble) on *test_df*.

    :param test_df: DataFrame with columns ``bacterium_embedding``,
        ``phage_embedding``, and ``interaction_type``.
    :param model: Trained model, a **list** of models (ensemble), or a
        :class:`SklearnClassifier`. When a list is passed, predictions are
        the averaged softmax of all members.
    :param batch_size: Batch size (used for NN models only).
    :param device: Device string (e.g. ``"cpu"`` or ``"cuda:0"``).
    :param threshold: Decision threshold applied to ensemble-averaged probabilities.
        Only used when *model* is a list. Defaults to 0.5.
    :param silent: Suppress log output.
    :return: ``(confusion_matrix, test_loss)``.  ``test_loss`` is ``-1`` for sklearn
        models or when using ensemble (loss is not aggregated).
        Confusion matrix format: ``[[tn, fp], [fn, tp]]``.
    """
    if isinstance(model, list):
        return test_nn_ensemble(test_df, model, batch_size, device, threshold, silent)
    elif isinstance(model, nn.Module):
        return test_nn_model(test_df, model, batch_size, device, threshold, silent)
    else:
        return test_sklearn_model(test_df, model, batch_size, device, threshold, silent)


def get_ensemble_probabilities(
    test_df: pd.DataFrame,
    ensemble: List[Any],
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble-averaged positive-class probabilities for every sample.

    :param test_df: Test DataFrame.
    :param ensemble: List of trained models (nn.Module or SklearnClassifier).
    :param batch_size: Batch size.
    :param device: Device string.
    :return: ``(probabilities, labels)`` where *probabilities* is a 1-D array of
        ensemble-averaged probabilities and *labels* is the ground-truth integer array.
    """
    if not ensemble:
        raise ValueError("Ensemble list cannot be empty.")

    if not isinstance(ensemble[0], nn.Module):
        X_test, y_test = dataframe_to_numpy_X_y(test_df)
        member_probs = []
        for model in ensemble:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "sklearn_model") and hasattr(model.sklearn_model, "predict_proba"):
                probs = model.sklearn_model.predict_proba(X_test)[:, 1]
            else:
                probs = model.predict(X_test).astype(float)
            member_probs.append(probs)
        avg_probs = np.mean(member_probs, axis=0)
        return avg_probs, np.asarray(y_test)

    dataloader = dataframe_to_tf_dataloader(test_df, batch_size, device)
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for model in ensemble:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                bact_emb, phg_emb, kmer_emb, labels = batch
            else:
                bact_emb, phg_emb, labels = batch
                kmer_emb = None
            member_probs = []
            for model in ensemble:
                logits = model(bact_emb, phg_emb, kmer_emb)
                probs = F.softmax(logits, dim=-1)[:, 1]  # positive class probability
                member_probs.append(probs)
            # Average across ensemble members directly on GPU before copying to CPU
            avg_probs = torch.stack(member_probs, dim=0).mean(dim=0)
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def test_nn_ensemble(
    test_df: pd.DataFrame,
    ensemble: List[nn.Module],
    batch_size: int,
    device: str,
    threshold: float = 0.5,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Evaluate a list of PyTorch models as an ensemble on *test_df*.

    For each batch, runs forward passes through all models, averages the
    softmax ``[:, 1]`` (positive-class) probabilities, and applies *threshold*
    to obtain predictions.

    :param test_df: Test DataFrame.
    :param ensemble: List of trained ``nn.Module`` models.
    :param batch_size: Batch size.
    :param device: Device string.
    :param threshold: Decision threshold for the positive class.  Defaults to
        0.5.  Use a lower value (e.g. 0.01) to boost recall at the cost of
        precision.
    :param silent: Suppress log output.
    :return: ``(confusion_matrix, -1)`` — ensemble loss is not computed.
        Confusion matrix format: ``[[tn, fp], [fn, tp]]``.
    """
    if not silent:
        logger.info(
            f"Starting ensemble testing ({len(ensemble)} members, threshold={threshold})..."
        )

    probs, labels = get_ensemble_probabilities(test_df, ensemble, batch_size, device)
    predictions = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel().tolist()

    if not silent:
        acc, rec, f1 = compute_metrics(tn, fp, fn, tp)
        logger.info(f"Accuracy (ensemble test): {acc:.4f}")
        logger.info(f"Recall (ensemble test): {rec:.4f}")
        logger.info(f"F1 score (ensemble test): {f1:.4f}")
        logger.info(
            f"Confusion Matrix (ensemble test) (TP, FP, FN, TN): {tp, fp, fn, tn}"
        )

    return cm, -1.0


def test_nn_model(
    test_df: pd.DataFrame,
    model: nn.Module,
    batch_size: int,
    device: str,
    threshold: float = 0.5,
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

    test_loss = torch.tensor(0.0, device=device)
    model.to(device)
    model.eval()
    accuracy.reset()
    f1.reset()
    recall.reset()
    cm_metric.reset()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                bact_emb, phg_emb, kmer_emb, labels = batch
            else:
                bact_emb, phg_emb, labels = batch
                kmer_emb = None
            logits = model(bact_emb, phg_emb, kmer_emb)
            loss = criterion(logits, labels)
            test_loss += loss.detach() * bact_emb.size(0)

            if threshold != 0.5:
                probs = F.softmax(logits, dim=1)[:, 1]
                predictions = (probs >= threshold).long()
            else:
                predictions = logits.argmax(dim=1, keepdim=True).squeeze()

            if not silent:
                accuracy.update(predictions, labels)
                f1.update(predictions, labels)
                recall.update(predictions, labels)
            cm_metric.update(predictions, labels)

    cm_mat = cm_metric.compute().cpu().numpy()  # torchmetrics default: TN, FP, FN, TP
    test_loss = (test_loss / len(dataloader.dataset)).item()  # type: ignore
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
    threshold: float = 0.5,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Evaluate a SklearnClassifier on *test_df*.

    See :func:`test_model` for parameter descriptions.
    """
    if not silent:
        logger.info("Starting testing...")

    X_test, y_test = dataframe_to_numpy_X_y(test_df)
    if threshold != 0.5 and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
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
