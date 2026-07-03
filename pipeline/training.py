"""
Model training helpers.

Provides a single public ``train_model()`` dispatcher and the underlying
``train_nn_model()`` / ``_train_sklearn_model()`` implementations, plus
``kfold_train()`` for K-Fold Cross Validation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedGroupKFold
from tqdm import tqdm
from typing import Literal

from pbi_models.classifiers.sklearn_classifier import SklearnClassifier
from pbi_utils.config_parser import TrainingConfig
from pbi_utils.logging import Logging
from pipeline.data import dataframe_to_tf_dataloader, dataframe_to_numpy_X_y
from pipeline.evaluation import test_model, compute_metrics

logger = Logging()

def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build and return the LR scheduler specified by ``training_config.lr_schedule``.

    :param optimizer: The optimizer to attach the scheduler to.
    :param training_config: Training configuration.
    :return: A configured scheduler, or ``None`` when ``lr_schedule="none"``.
    """
    schedule = training_config.lr_schedule

    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs, eta_min=1e-6
        )
    elif schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=(
                "min" if training_config.monitor_metric_reduce_lr == "loss" else "max"
            ),
            factor=training_config.multiplying_factor_reduce_lr,
            patience=training_config.patience_reduce_lr,
        )
    elif schedule == "none":
        return None
    else:
        raise ValueError(
            f"Unknown lr_schedule {schedule!r}. "
            "Allowed values: 'plateau', 'cosine', 'none'."
        )


def train_model(
    train_df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
    pos_weight: float = 1.0,
) -> np.ndarray:
    """
    Train a model (PyTorch nn.Module or SklearnClassifier) on *train_df*.

    Dispatches internally to :func:`train_nn_model` or
    :func:`_train_sklearn_model` based on the model type.

    :param train_df: Training DataFrame with columns ``bacterium_embedding``,
        ``phage_embedding``, and ``interaction_type``.
    :param model: Model to train.
    :param training_config: Training hyper-parameters.
    :param device: Device string (e.g. ``"cpu"`` or ``"cuda:0"``).
    :param val_df: Optional validation DataFrame.  Required for early stopping
        and ``lr_schedule="plateau"``.
    :param verbose: Verbosity level (0 = silent, 1 = progress bar, 2 = full).
    :param progressbar_description: Label shown on the tqdm progress bar.
    :param pos_weight: Class weight for the positive class (used by NN only).
    :return: Training confusion matrix in ``[[tn, fp], [fn, tp]]`` format.
    """
    if isinstance(model, nn.Module):
        return train_nn_model(
            train_df,
            model,
            training_config,
            device,
            val_df,
            verbose,
            progressbar_description,
            pos_weight,
        )
    else:
        return _train_sklearn_model(
            train_df,
            model,
            training_config,
            device,
            val_df,
            verbose,
            progressbar_description,
        )


def train_nn_model(
    train_df: pd.DataFrame,
    model: nn.Module,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
    pos_weight: float = 1.0,
) -> np.ndarray:
    """
    Train a PyTorch nn.Module on *train_df*.

    See :func:`train_model` for parameter descriptions.

    The LR scheduler is selected by ``training_config.lr_schedule``:
    - ``"plateau"`` advances on each validation step.
    - ``"cosine"`` advances once per epoch (no validation set required).
    - ``"none"`` disables scheduling.
    """
    dataloader = dataframe_to_tf_dataloader(
        train_df, batch_size=training_config.batch_size, device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    class_weights = torch.tensor([1.0, float(pos_weight)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = _create_scheduler(optimizer, training_config)

    # Metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm_metric = tm.ConfusionMatrix(task="binary").to(device)

    if verbose >= 2:
        logger.info(f"Starting training for {training_config.epochs} epochs...")

    # Early stopping state
    best_metric = (
        -np.inf
        if training_config.monitor_metric_early_stopping != "loss"
        else np.inf
    )
    epochs_no_improve = 0
    best_model_state = None

    with tqdm(
        range(training_config.epochs),
        unit="epoch",
        desc=progressbar_description,
        disable=verbose < 1,
    ) as tepochs:
        for epoch in tepochs:
            model.train()
            train_loss = 0.0
            for bact_emb, phg_emb, labels in dataloader:
                optimizer.zero_grad()

                # Optional embedding noise for regularisation
                if training_config.training_noise_std != 0:
                    bact_emb = bact_emb + (
                        torch.randn_like(bact_emb) * training_config.training_noise_std
                    )
                    phg_emb = phg_emb + (
                        torch.randn_like(phg_emb) * training_config.training_noise_std
                    )

                logits = model(bact_emb, phg_emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bact_emb.size(0)

                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                acc = accuracy(predictions, labels)
                f1s = f1(predictions, labels)
                rec = recall(predictions, labels)
                cm_metric(predictions, labels)

            # Advance non-plateau schedulers once per epoch
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            # Validation step
            if val_df is not None:
                val_cm, val_loss = test_model(
                    val_df, model, training_config.batch_size, device, silent=True
                )
                model.train()

                tn, fp, fn, tp = val_cm.ravel().tolist()
                val_acc, val_rec, val_f1 = compute_metrics(tn, fp, fn, tp)

                # Early stopping
                if training_config.monitor_metric_early_stopping == "f1":
                    current_metric = val_f1
                    improved = current_metric > best_metric
                elif training_config.monitor_metric_early_stopping == "loss":
                    current_metric = val_loss
                    improved = current_metric < best_metric
                else:
                    raise ValueError(
                        f"Unknown metric {training_config.monitor_metric_early_stopping!r}. "
                        "monitor_metric_early_stopping must be 'f1' or 'loss'."
                    )

                if improved:
                    best_metric = current_metric
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= training_config.patience_early_stopping:
                    if verbose >= 1:
                        logger.debug(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Advance plateau scheduler on validation F1
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_f1)

                tepochs.set_postfix(
                    OrderedDict(
                        lr=optimizer.param_groups[0]["lr"],
                        loss=val_loss,
                        accuracy=100.0 * val_acc,
                        recall=100.0 * val_rec,
                        f1=100.0 * val_f1,
                    )
                )
            else:
                tepochs.set_postfix(
                    OrderedDict(
                        loss=loss.item(),
                        accuracy=100.0 * acc.item(),
                        recall=100.0 * rec.item(),
                        f1=100.0 * f1s.item(),
                    )
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    cm_mat = cm_metric.compute().cpu().numpy()  # torchmetrics default: TN, FP, FN, TP
    tn, fp, fn, tp = cm_mat.ravel().tolist()

    if verbose >= 2:
        logger.info("Finished training")
        logger.info(f"Accuracy (train): {accuracy.compute()}")
        logger.info(f"Recall (train): {recall.compute()}")
        logger.info(f"F1 score (train): {f1.compute()}")
        logger.info(f"Loss (train): {loss}")
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm_mat


def _train_sklearn_model(
    train_df: pd.DataFrame,
    model: SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
) -> np.ndarray:
    """
    Train a SklearnClassifier on *train_df*.

    See :func:`train_model` for parameter descriptions.
    """
    if verbose >= 2:
        logger.info("Starting training...")

    X_train, y_train = dataframe_to_numpy_X_y(train_df)
    model.fit(X_train, y_train)

    if val_df is not None:
        X_val, y_val = dataframe_to_numpy_X_y(val_df)
        y_pred = model.predict(X_val)
    else:
        y_val = y_train
        y_pred = model.predict(X_train)

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    if verbose >= 2:
        logger.info("Finished training")
        logger.info(f"Accuracy (train): {acc}")
        logger.info(f"Recall (train): {rec}")
        logger.info(f"F1 score (train): {f1}")
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm


def kfold_train(
    df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
) -> np.ndarray:
    """
    Perform K-Fold Cross Validation training.

    :param df: Full dataset DataFrame with columns ``bacterium_embedding``,
        ``phage_embedding``, ``interaction_type``, ``phage_id``.
    :param model: Model to train (must implement ``reset_model(device)``).
    :param training_config: Training hyper-parameters (uses ``k_folds_cv`` and
        ``stratify_cv``).
    :param device: Device string.
    :return: Mean confusion matrix across all folds (``[[tn, fp], [fn, tp]]``).
    """
    if training_config.stratify_cv:
        splitter = StratifiedGroupKFold(n_splits=training_config.k_folds_cv)
        groups = df["phage_id"].values
        y = df["interaction_type"].values
        splits = splitter.split(df, y=y, groups=groups)
    else:
        kfold = KFold(n_splits=training_config.k_folds_cv, shuffle=True, random_state=42)
        splits = kfold.split(df)

    all_conf_matrices = []
    logger.info(f"Starting {training_config.k_folds_cv}-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.debug(f"Starting fold {fold + 1}...")
        train_fold_df = df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = df.iloc[val_idx].reset_index(drop=True)

        num_neg = (train_fold_df["interaction_type"] == 0).sum()
        num_pos = (train_fold_df["interaction_type"] == 1).sum()
        curr_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        model.reset_model(device)
        train_model(
            train_df=train_fold_df,
            model=model,
            training_config=training_config,
            device=device,
            val_df=val_fold_df,
            verbose=1,
            progressbar_description=f"Fold {fold + 1}/{training_config.k_folds_cv}",
            pos_weight=curr_pos_weight,
        )

        cm_mat, _ = test_model(
            test_df=val_fold_df,
            model=model,
            batch_size=training_config.batch_size,
            device=device,
            silent=True,
        )
        all_conf_matrices.append(cm_mat)

    mean_cm: np.ndarray = sum(all_conf_matrices) / len(all_conf_matrices)
    tn, fp, fn, tp = mean_cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    logger.info("Finished Cross Validation training")
    logger.info(f"Accuracy (CV): {acc}")
    logger.info(f"Recall (CV): {rec}")
    logger.info(f"F1 score (CV): {f1}")
    logger.info(
        f"Confusion Matrix (CV) (TP, FP, FN, TN): "
        f"({tp:.2f}, {fp:.2f}, {fn:.2f}, {tn:.2f})"
    )

    return mean_cm
