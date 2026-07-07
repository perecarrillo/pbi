"""
Training/testing run statistics.

The ``Stats`` class collects confusion matrices, timing information, and run
metadata and serialises them as a tab-separated line compatible with the Google Sheets format.
"""

from __future__ import annotations

import datetime
import json
from typing import Callable

import numpy as np
import torch.nn as nn

from pbi_utils.logging import Logging

logger = Logging()


class Stats:
    """
    Accumulate training and testing statistics for a single pipeline run.

    Call :meth:`update_classifier`, :meth:`update_ensemble_info`,
    :meth:`update_train_results`, and :meth:`update_test_results` as the
    pipeline progresses, then call :meth:`log` to serialise the results.
    """

    def __init__(self, config) -> None:
        """
        :param config: The :class:`~pbi_utils.config_parser.Config` object for
            the current run.  Used to retrieve model names and hyper-parameters
            when logging.
        """
        self.config = config
        self.test_cm: np.ndarray | None = None  # always (TN, FP, FN, TP)
        self.train_cm: np.ndarray | None = None
        self.classifier: nn.Module | None = None
        self.train_time: float | None = None
        self.test_time: float | None = None

        self.ensemble_size: int = getattr(config, "ensemble_size", 1)
        self.seed: int = getattr(config, "seed", 42)
        self.split_strategy: str = config.training_config.test_split_strategy
        self.threshold: float = 0.5
        self.calibrated_threshold: float | None = None

    def update_test_results(self, cm: np.ndarray, test_time: float) -> None:
        """Record the test confusion matrix and elapsed test time (seconds)."""
        self.test_cm = cm
        self.test_time = test_time

    def update_train_results(self, cm: np.ndarray, train_time: float) -> None:
        """Record the training confusion matrix and elapsed training time (seconds)."""
        self.train_cm = cm
        self.train_time = train_time

    def update_classifier(self, classifier: nn.Module) -> None:
        """Record the instantiated classifier for name-based logging."""
        self.classifier = classifier

    def update_ensemble_info(
        self,
        ensemble_size: int,
        seed: int,
        split_strategy: str,
        threshold: float,
        calibrated_threshold: float | None,
    ) -> None:
        """Record ensemble and threshold metadata."""
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.split_strategy = split_strategy
        self.threshold = threshold
        self.calibrated_threshold = calibrated_threshold

    def log(self, write_fn: Callable[[str], None]) -> None:
        """
        Serialise the statistics as a TSV line and pass it to *write_fn*.

        The line format matches the Google Sheet tracking columns used by the
        project.

        :param write_fn: A callable that receives the formatted string, e.g.
            ``lambda msg: f.write(msg + "\\n")``.
        """
        bacteria_model_names = [
            str(x) for x in self.config.bacteria_embedding_models
        ]
        phages_model_names = [
            str(x) for x in self.config.phages_embedding_models
        ]

        def _fmt(value: float | None) -> str:
            return f"{value:.4f}".replace(".", ",") if value is not None else ""

        def _cm_metrics(cm: np.ndarray | None):
            """Return (acc, rec, f1) strings from a confusion matrix, or empty strings."""
            if cm is None:
                return "", "", ""
            tn, fp, fn, tp = (
                float(cm[0][0]), float(cm[0][1]),
                float(cm[1][0]), float(cm[1][1]),
            )
            denom_acc = tp + tn + fp + fn
            acc = (tp + tn) / denom_acc if denom_acc > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            return _fmt(acc), _fmt(rec), _fmt(f1)

        train_acc, train_rec, train_f1 = _cm_metrics(self.train_cm)
        test_acc, test_rec, test_f1 = _cm_metrics(self.test_cm)

        data = [
            datetime.datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
            f"main.py -j '{json.dumps(self.config.raw_dict)}'",
            "",  # Description column (left blank; fill manually)
            ", ".join(bacteria_model_names),
            ", ".join(phages_model_names),
            self.classifier.name() if self.classifier is not None else "",
            str(self.config.training_config.epochs),
            str(self.config.training_config.batch_size),
            f"{self.config.training_config.learning_rate:.4e}".replace(".", ","),
            str(self.ensemble_size),
            self.split_strategy,
            f"{self.threshold:.2f}".replace(".", ","),
            _fmt(self.train_time),
            # Train confusion matrix: TP, FP, FN, TN
            _fmt(self.train_cm[1][1] if self.train_cm is not None else None),
            _fmt(self.train_cm[0][1] if self.train_cm is not None else None),
            _fmt(self.train_cm[1][0] if self.train_cm is not None else None),
            _fmt(self.train_cm[0][0] if self.train_cm is not None else None),
            train_acc,
            train_rec,
            train_f1,
            _fmt(self.test_time),
            # Test confusion matrix: TP, FP, FN, TN
            _fmt(self.test_cm[1][1] if self.test_cm is not None else None),
            _fmt(self.test_cm[0][1] if self.test_cm is not None else None),
            _fmt(self.test_cm[1][0] if self.test_cm is not None else None),
            _fmt(self.test_cm[0][0] if self.test_cm is not None else None),
            test_acc,
            test_rec,
            test_f1,
        ]

        write_fn("\n" + "\t".join(data))
