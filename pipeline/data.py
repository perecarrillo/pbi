"""
Dataset construction and DataLoader helpers.

Functions for loading embeddings from the EmbeddingsManager, assembling a unified
training/testing DataFrame, and converting it to PyTorch DataLoaders or numpy arrays.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List

from pbi_utils.data_manager import EmbeddingsManager
from pbi_utils.logging import Logging

logger = Logging()


def make_dataset(
    couples_df: pd.DataFrame,
    bacteria_model_names: List[str],
    phages_model_names: List[str],
    output_manager: EmbeddingsManager,
    device: str,
) -> pd.DataFrame:
    """
    Create a dataset with embeddings for bacteria and phages by loading them from
    the output manager. The final DataFrame will have columns: bacterium_id, phage_id,
    interaction_type, bacterium_embedding, phage_embedding.

    Multiple model embeddings are **concatenated** into one final embedding per organism.

    :param couples_df: DataFrame with columns ``bacterium_id``, ``phage_id``,
        ``interaction_type``.
    :param bacteria_model_names: Names of bacteria embedding models to load.
    :param phages_model_names: Names of phage embedding models to load.
    :param output_manager: EmbeddingsManager used to load stored embeddings.
    :param device: Device to load tensors onto (e.g. ``"cpu"`` or ``"cuda:0"``).
    :return: DataFrame with columns ``bacterium_id``, ``phage_id``,
        ``interaction_type``, ``bacterium_embedding``, ``phage_embedding``.
    """
    result = couples_df.copy(deep=True)

    logger.info("Creating dataset (loading embeddings)...")

    # Load and concatenate all bacteria embeddings
    bacteria_embeddings = [
        output_manager.load_embedding_batch(
            result["bacterium_id"].tolist(),
            model_name=model_name,
            device=device,
        )
        for model_name in bacteria_model_names
    ]
    result["bacterium_embedding"] = pd.Series(
        [torch.cat(embeds) for embeds in zip(*bacteria_embeddings)]
    )

    # Load and concatenate all phage embeddings
    phage_embeddings = [
        output_manager.load_embedding_batch(
            result["phage_id"].tolist(),
            model_name=model_name,
            device=device,
        )
        for model_name in phages_model_names
    ]
    result["phage_embedding"] = pd.Series(
        [torch.cat(embeds) for embeds in zip(*phage_embeddings)]
    )

    logger.debug(
        f"Final embedding size (bacteria): {len(result['bacterium_embedding'].iloc[0])}"
    )
    logger.debug(
        f"Final embedding size (phages): {len(result['phage_embedding'].iloc[0])}"
    )

    return result


def dataframe_to_tf_dataloader(
    df: pd.DataFrame, batch_size: int, device: str
) -> DataLoader:
    """
    Convert a DataFrame with embeddings and interaction types into a PyTorch DataLoader.

    :param df: DataFrame with columns ``bacterium_embedding``, ``phage_embedding``,
        and ``interaction_type``.
    :param batch_size: Batch size for the DataLoader.
    :param device: Device to load tensors onto (e.g. ``"cpu"`` or ``"cuda:0"``).
    :return: PyTorch DataLoader yielding ``(bact_emb, phg_emb, labels)`` batches.
    """
    dataset = TensorDataset(
        torch.stack(list(df["bacterium_embedding"])).to(device),
        torch.stack(list(df["phage_embedding"])).to(device),
        torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
    )
    return DataLoader(dataset, batch_size=batch_size)


def dataframe_to_numpy_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a DataFrame with embeddings and interaction types into numpy arrays.

    Bacteria and phage embeddings are concatenated into a single feature vector per row.

    :param df: DataFrame with columns ``bacterium_embedding``, ``phage_embedding``,
        and ``interaction_type``.
    :return: Tuple ``(X, y)`` where X contains the concatenated feature vectors and
        y contains the interaction type labels.
    """
    y = df["interaction_type"]
    X = df[["bacterium_embedding", "phage_embedding"]]

    X = X.apply(
        lambda x: np.concatenate(
            [x["bacterium_embedding"].cpu().numpy(), x["phage_embedding"].cpu().numpy()],
            axis=None,
        ),
        axis=1,
        result_type="expand",
    )
    return X, y
