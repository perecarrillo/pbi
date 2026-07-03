"""
Dimensionality reduction for organism embeddings.

The public surface is three functions:

* ``fit_pca``: fit PCA on a dataset (training set only) and return the fitted objects.
* ``transform_pca``: apply previously fitted PCA objects to any dataset (train or test).
* ``reduce_dimensionality``: high-level wrapper; dispatches by technique and returns
  both the transformed dataset **and** the fitted reducer objects so the caller can
  reuse them (e.g. to transform a held-out test set with the same PCA).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from pbi_utils.logging import Logging
from pbi_utils.types import DIMENSIONALITY_REDUCTION_TECHNIQUE

logger = Logging()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _plot_explained_variance(
    exp_var: np.ndarray, output_path: str, title: str = "PCA Explained Variance"
) -> None:
    """Save a cumulative-explained-variance step plot to *output_path*."""
    cum_sum = np.cumsum(exp_var)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.step(
        range(1, len(cum_sum) + 1),
        cum_sum,
        where="mid",
        label="Cumulative explained variance",
    )
    ax.set_ylabel("Explained variance")
    ax.set_xlabel("nº of components")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _embeddings_to_numpy(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of torch tensors to a 2-D numpy array."""
    return np.stack(
        [x.detach().cpu().numpy() for x in series], axis=0
    )


def _numpy_to_tensor_list(array: np.ndarray) -> list:
    """Convert a 2-D numpy array back to a list of 1-D float32 tensors."""
    return list(torch.from_numpy(array).float())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_pca(
    dataset: pd.DataFrame,
    n_components_bact: int | None,
    n_components_phag: int | None,
    output_dir: str | None = None,
    random_state: int = 42,
) -> Tuple[PCA, PCA]:
    """
    Fit PCA on the bacterium and phage embeddings of *dataset*.

    Call this on the **training set only** to avoid data leakage from the test set.
    Pass the returned PCA objects to :func:`transform_pca` to apply them to any split.

    :param dataset: DataFrame with ``bacterium_embedding`` and ``phage_embedding``
        columns containing torch tensors.
    :param n_components_bact: Number of PCA components to keep for bacteria.
        ``None`` keeps all components.
    :param n_components_phag: Number of PCA components to keep for phages.
        ``None`` keeps all components.
    :param output_dir: If provided, saves explained-variance plots there.
    :param random_state: Random seed for PCA reproducibility.
    :return: ``(pca_bact, pca_phag)``, fitted sklearn PCA objects.
    """
    pca_bact = PCA(random_state=random_state, n_components=n_components_bact)
    pca_phag = PCA(random_state=random_state, n_components=n_components_phag)

    pca_bact.fit(_embeddings_to_numpy(dataset["bacterium_embedding"]))
    pca_phag.fit(_embeddings_to_numpy(dataset["phage_embedding"]))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _plot_explained_variance(
            pca_bact.explained_variance_ratio_,
            os.path.join(output_dir, "pca_explained_variance_bacterium.png"),
            "PCA Explained Variance (Bacterium Embedding)",
        )
        _plot_explained_variance(
            pca_phag.explained_variance_ratio_,
            os.path.join(output_dir, "pca_explained_variance_phage.png"),
            "PCA Explained Variance (Phage Embedding)",
        )

    return pca_bact, pca_phag


def transform_pca(
    dataset: pd.DataFrame,
    pca_bact: PCA,
    pca_phag: PCA,
) -> pd.DataFrame:
    """
    Apply *fitted* PCA objects to the embeddings in *dataset* (in-place copy).

    Use the PCA objects returned by :func:`fit_pca`.  Never call this with PCA objects
    fitted on test data.

    :param dataset: DataFrame with ``bacterium_embedding`` and ``phage_embedding``
        columns containing torch tensors.
    :param pca_bact: Fitted PCA for bacteria embeddings.
    :param pca_phag: Fitted PCA for phage embeddings.
    :return: New DataFrame with reduced-dimensionality embeddings.
    """
    result = dataset.copy()

    result["bacterium_embedding"] = _numpy_to_tensor_list(
        pca_bact.transform(_embeddings_to_numpy(result["bacterium_embedding"]))
    )
    result["phage_embedding"] = _numpy_to_tensor_list(
        pca_phag.transform(_embeddings_to_numpy(result["phage_embedding"]))
    )

    logger.debug(
        f"Embedding size after PCA (bacteria): "
        f"{len(result['bacterium_embedding'].iloc[0])}"
    )
    logger.debug(
        f"Embedding size after PCA (phages): "
        f"{len(result['phage_embedding'].iloc[0])}"
    )

    return result


def reduce_dimensionality(
    dataset: pd.DataFrame,
    technique: DIMENSIONALITY_REDUCTION_TECHNIQUE,
    output_dir: str | None,
    n_components_bact: int | None = None,
    n_components_phag: int | None = None,
) -> Tuple[pd.DataFrame, PCA | None, PCA | None]:
    """
    High-level dimensionality reduction wrapper.

    Dispatches to the appropriate technique and returns both the transformed dataset
    **and** the fitted reducer objects.  Callers that need to transform a separate
    test set should pass the returned PCA objects to :func:`transform_pca`.

    :param dataset: DataFrame with ``bacterium_embedding`` and ``phage_embedding``
        columns containing torch tensors.
    :param technique: Dimensionality reduction technique.  One of
        ``DIMENSIONALITY_REDUCTION_TECHNIQUE`` (``"none"`` or ``"PCA"``).
    :param output_dir: Directory for output plots (PCA only).
    :param n_components_bact: Number of PCA components for bacteria (``None`` = all).
    :param n_components_phag: Number of PCA components for phages (``None`` = all).
    :return: ``(transformed_dataset, pca_bact, pca_phag)``.  The PCA objects are
        ``None`` when technique is ``"none"``.
    :raises ValueError: For unsupported techniques.
    """
    if technique == "none":
        return dataset, None, None

    elif technique == "PCA":
        pca_bact, pca_phag = fit_pca(
            dataset, n_components_bact, n_components_phag, output_dir
        )
        transformed = transform_pca(dataset, pca_bact, pca_phag)
        return transformed, pca_bact, pca_phag

    else:
        raise ValueError(
            f"{technique!r} is not a supported dimensionality reduction technique. "
            f"Allowed values: {DIMENSIONALITY_REDUCTION_TECHNIQUE}"
        )
