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
        """Record ensemble metadata (split_strategy and threshold kept for compatibility)."""
        self.ensemble_size = ensemble_size
        self.seed = seed

    def log(self, write_fn: Callable[[str], None]) -> None:
        """
        Serialise the statistics as a TSV line and pass it to *write_fn*.

        The line format matches the Google Sheet tracking columns used by the
        project.

        :param write_fn: A callable that receives the formatted string, e.g.
            ``lambda msg: f.write(msg + "\\n")``.
        """
        bacteria_model_names = [str(x) for x in self.config.bacteria_embedding_models]
        phages_model_names = [str(x) for x in self.config.phages_embedding_models]

        def _fmt(value: float | None) -> str:
            return f"{value:.4f}".replace(".", ",") if value is not None else ""

        def _cm_fields(cm: np.ndarray | None, include_specificity: bool = False) -> list:
            """Return formatted metric strings from a confusion matrix.

            Train order: TP, FP, FN, TN, Accuracy, Weighted Accuracy, Precision, Recall, F1, MCC
            Test order:  TP, FP, FN, TN, Accuracy, Weighted Accuracy, Precision, Recall, Specificity, F1, MCC
            """
            if cm is None:
                return [""] * (11 if include_specificity else 10)
            tn, fp, fn, tp = (
                float(cm[0][0]), float(cm[0][1]),
                float(cm[1][0]), float(cm[1][1]),
            )
            total = tp + tn + fp + fn
            acc   = (tp + tn) / total if total > 0 else 0.0
            rec   = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
            spec  = tn / (tn + fp)    if (tn + fp) > 0 else 0.0
            w_acc = (rec + spec) / 2
            prec  = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
            f1    = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            mcc   = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

            fields = [_fmt(tp), _fmt(fp), _fmt(fn), _fmt(tn),
                      _fmt(acc), _fmt(w_acc), _fmt(prec), _fmt(rec)]
            if include_specificity:
                fields.append(_fmt(spec))
            fields += [_fmt(f1), _fmt(mcc)]
            return fields

        tc = self.config.training_config

        data = [
            datetime.datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
            f"main.py -j '{json.dumps(self.config.raw_dict)}'",
            "",  # Description (left blank; fill manually)
            ", ".join(bacteria_model_names),
            ", ".join(phages_model_names),
            self.classifier.name() if self.classifier is not None else "",
            str(tc.epochs),
            str(tc.batch_size),
            f"{tc.learning_rate:.4e}".replace(".", ","),
            f"{tc.weight_decay:.4e}".replace(".", ","),
            f"{tc.training_noise_std:.4f}".replace(".", ","),
            str(self.ensemble_size),
            # Train section
            _fmt(self.train_time),
            *_cm_fields(self.train_cm, include_specificity=False),
            # Test section
            _fmt(self.test_time),
            *_cm_fields(self.test_cm, include_specificity=True),
        ]

        write_fn("\n" + "\t".join(data))
