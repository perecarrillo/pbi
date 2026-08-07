from __future__ import annotations

import os
from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # avoids tkinter threading crashes
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


def fit_pca(
    dataset: pd.DataFrame,
    n_components_bact: int | None,
    n_components_phag: int | None,
    output_dir: str | None = None,
    random_state: int = 42,
) -> Tuple[PCA | None, PCA | None]:
    bact_dim = len(dataset["bacterium_embedding"].iloc[0]) if len(dataset) > 0 else 0
    phag_dim = len(dataset["phage_embedding"].iloc[0]) if len(dataset) > 0 else 0

    pca_bact = (
        PCA(random_state=random_state, n_components=n_components_bact)
        if (bact_dim > 0 and n_components_bact is not None)
        else None
    )
    pca_phag = (
        PCA(random_state=random_state, n_components=n_components_phag)
        if (phag_dim > 0 and n_components_phag is not None)
        else None
    )

    if pca_bact is not None:
        pca_bact.fit(_embeddings_to_numpy(dataset["bacterium_embedding"]))
    if pca_phag is not None:
        pca_phag.fit(_embeddings_to_numpy(dataset["phage_embedding"]))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if pca_bact is not None:
            _plot_explained_variance(
                pca_bact.explained_variance_ratio_,
                os.path.join(output_dir, "pca_explained_variance_bacterium.png"),
                "PCA Explained Variance (Bacterium Embedding)",
            )
        if pca_phag is not None:
            _plot_explained_variance(
                pca_phag.explained_variance_ratio_,
                os.path.join(output_dir, "pca_explained_variance_phage.png"),
                "PCA Explained Variance (Phage Embedding)",
            )

    return pca_bact, pca_phag


def transform_pca(
    dataset: pd.DataFrame,
    pca_bact: PCA | None,
    pca_phag: PCA | None,
) -> pd.DataFrame:
    """
    Apply *fitted* PCA objects to the embeddings in *dataset* (in-place copy).

    Use the PCA objects returned by :func:`fit_pca`.  Never call this with PCA objects
    fitted on test data.

    :param dataset: DataFrame with ``bacterium_embedding`` and ``phage_embedding``
        columns containing torch tensors.
    :param pca_bact: Fitted PCA for bacteria embeddings (or None).
    :param pca_phag: Fitted PCA for phage embeddings (or None).
    :return: New DataFrame with reduced-dimensionality embeddings.
    """
    result = dataset.copy()

    if pca_bact is not None:
        result["bacterium_embedding"] = _numpy_to_tensor_list(
            pca_bact.transform(_embeddings_to_numpy(result["bacterium_embedding"]))
        )
    if pca_phag is not None:
        result["phage_embedding"] = _numpy_to_tensor_list(
            pca_phag.transform(_embeddings_to_numpy(result["phage_embedding"]))
        )

    bact_dim = len(result["bacterium_embedding"].iloc[0]) if len(result) > 0 else 0
    phage_dim = len(result["phage_embedding"].iloc[0]) if len(result) > 0 else 0

    logger.debug(f"Embedding size after PCA (bacteria): {bact_dim}")
    logger.debug(f"Embedding size after PCA (phages): {phage_dim}")

    return result


# ---------------------------------------------------------------------------
# UMAP helpers
# ---------------------------------------------------------------------------


def fit_umap(
    dataset: pd.DataFrame,
    n_components: int = 200,
    random_state: int = 42,
) -> Tuple[Any, Any]:
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP reduction. Install it with: "
            "pip install umap-learn"
        ) from e

    logger.info(f"Fitting UMAP (n_components={n_components}) on bacteria embeddings...")
    umap_bact = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_jobs=1,  # deterministic
    )
    umap_bact.fit(_embeddings_to_numpy(dataset["bacterium_embedding"]))

    logger.info(f"Fitting UMAP (n_components={n_components}) on phage embeddings...")
    umap_phag = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_jobs=1,
    )
    umap_phag.fit(_embeddings_to_numpy(dataset["phage_embedding"]))

    return umap_bact, umap_phag


def transform_umap(
    dataset: pd.DataFrame,
    umap_bact: Any,
    umap_phag: Any,
) -> pd.DataFrame:
    result = dataset.copy()

    result["bacterium_embedding"] = _numpy_to_tensor_list(
        umap_bact.transform(_embeddings_to_numpy(result["bacterium_embedding"]))
    )
    result["phage_embedding"] = _numpy_to_tensor_list(
        umap_phag.transform(_embeddings_to_numpy(result["phage_embedding"]))
    )

    return result


def reduce_dimensionality(
    dataset: pd.DataFrame,
    technique: DIMENSIONALITY_REDUCTION_TECHNIQUE,
    output_dir: str | None,
    n_components_bact: int | None = None,
    n_components_phag: int | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Any | None, Any | None]:
    if technique == "none":
        return dataset, None, None

    elif technique == "PCA":
        pca_bact, pca_phag = fit_pca(
            dataset, n_components_bact, n_components_phag, output_dir, random_state
        )
        transformed = transform_pca(dataset, pca_bact, pca_phag)
        return transformed, pca_bact, pca_phag

    elif technique == "UMAP":
        n_comp = n_components_bact if n_components_bact is not None else 200
        umap_bact, umap_phag = fit_umap(dataset, n_components=n_comp, random_state=random_state)
        transformed = transform_umap(dataset, umap_bact, umap_phag)
        return transformed, umap_bact, umap_phag

    else:
        raise ValueError(
            f"{technique!r} is not a supported dimensionality reduction technique. "
            f"Allowed values: {DIMENSIONALITY_REDUCTION_TECHNIQUE}"
        )
