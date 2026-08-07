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

from pbi_utils.config_parser import KmerConfig
from pbi_utils.data_manager import EmbeddingsManager
from pbi_utils.logging import Logging
from pipeline.kmer import compute_kmer_features

logger = Logging()


def make_dataset(
    couples_df: pd.DataFrame,
    bacteria_model_names: List[str],
    phages_model_names: List[str],
    output_manager: EmbeddingsManager,
    device: str,
    kmer_config: KmerConfig | None = None,
    bacteria_df: pd.DataFrame | None = None,
    phages_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create a dataset with embeddings for bacteria and phages by loading them from
    the output manager. The final DataFrame will have columns: bacterium_id, phage_id,
    interaction_type, bacterium_embedding, phage_embedding, and optionally kmer_embedding.

    Multiple model embeddings are **concatenated** into one final embedding per organism.
    If kmer_config is provided along with sequence DataFrames, relative k-mer frequencies
    are computed and concatenated into kmer_embedding per pair.

    :param couples_df: DataFrame with columns ``bacterium_id``, ``phage_id``,
        ``interaction_type``.
    :param bacteria_model_names: Names of bacteria embedding models to load.
    :param phages_model_names: Names of phage embedding models to load.
    :param output_manager: EmbeddingsManager used to load stored embeddings.
    :param device: Device to load tensors onto (e.g. ``"cpu"`` or ``"cuda:0"``).
    :param kmer_config: Optional KmerConfig specifying k values and sequence columns.
    :param bacteria_df: Optional DataFrame with bacteria sequences.
    :param phages_df: Optional DataFrame with phage sequences.
    :return: DataFrame with columns ``bacterium_id``, ``phage_id``,
        ``interaction_type``, ``bacterium_embedding``, ``phage_embedding``, plus
        ``kmer_embedding`` if kmer_config is provided.
    """
    result = couples_df.copy(deep=True)

    logger.info("Creating dataset (loading embeddings)...")

    # Load and concatenate all bacteria embeddings
    if bacteria_model_names:
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
    else:
        result["bacterium_embedding"] = pd.Series(
            [torch.empty(0, device=device) for _ in range(len(result))]
        )

    # Load and concatenate all phage embeddings
    if phages_model_names:
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
    else:
        result["phage_embedding"] = pd.Series(
            [torch.empty(0, device=device) for _ in range(len(result))]
        )

    bact_dim = len(result["bacterium_embedding"].iloc[0]) if len(result) > 0 else 0
    phage_dim = len(result["phage_embedding"].iloc[0]) if len(result) > 0 else 0
    logger.debug(f"Final embedding size (bacteria): {bact_dim}")
    logger.debug(f"Final embedding size (phages): {phage_dim}")

    if kmer_config is not None and kmer_config.k_values:
        if bacteria_df is None or phages_df is None:
            raise ValueError(
                "bacteria_df and phages_df must be provided to make_dataset when kmer_config is set."
            )
        logger.info(f"Computing k-mer features (k={kmer_config.k_values})...")
        bact_col = "bacterium_sequence"
        phage_col = "phage_sequence"

        bact_map = (
            bacteria_df.set_index("bacterium_id")[bact_col].to_dict()
            if "bacterium_id" in bacteria_df.columns and bact_col in bacteria_df.columns
            else {}
        )
        phage_map = (
            phages_df.set_index("phage_id")[phage_col].to_dict()
            if "phage_id" in phages_df.columns and phage_col in phages_df.columns
            else {}
        )

        bact_cache = {}
        for b_id in result["bacterium_id"].unique():
            seq = str(bact_map.get(b_id, "")) if pd.notna(bact_map.get(b_id, "")) else ""
            feats = compute_kmer_features(seq, kmer_config.k_values)
            bact_cache[b_id] = torch.tensor(feats, dtype=torch.float32, device=device)

        phage_cache = {}
        for p_id in result["phage_id"].unique():
            seq = str(phage_map.get(p_id, "")) if pd.notna(phage_map.get(p_id, "")) else ""
            feats = compute_kmer_features(seq, kmer_config.k_values)
            phage_cache[p_id] = torch.tensor(feats, dtype=torch.float32, device=device)

        kmer_embeds = [
            torch.cat([bact_cache[b_id], phage_cache[p_id]], dim=0)
            for b_id, p_id in zip(result["bacterium_id"], result["phage_id"])
        ]
        result["kmer_embedding"] = pd.Series(kmer_embeds)
        logger.debug(
            f"Final k-mer vector size (bacteria + phages): {len(result['kmer_embedding'].iloc[0])}"
        )

    return result


def dataframe_to_tf_dataloader(
    df: pd.DataFrame, batch_size: int, device: str
) -> DataLoader:
    """
    Convert a DataFrame with embeddings and interaction types into a PyTorch DataLoader.

    :param df: DataFrame with columns ``bacterium_embedding``, ``phage_embedding``,
        ``interaction_type``, and optionally ``kmer_embedding``.
    :param batch_size: Batch size for the DataLoader.
    :param device: Device to load tensors onto (e.g. ``"cpu"`` or ``"cuda:0"``).
    :return: PyTorch DataLoader yielding ``(bact_emb, phg_emb, labels)`` or
        ``(bact_emb, phg_emb, kmer_emb, labels)`` batches.
    """
    if "kmer_embedding" in df.columns:
        dataset = TensorDataset(
            torch.stack(list(df["bacterium_embedding"])).to(device),
            torch.stack(list(df["phage_embedding"])).to(device),
            torch.stack(list(df["kmer_embedding"])).to(device),
            torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
        )
    else:
        dataset = TensorDataset(
            torch.stack(list(df["bacterium_embedding"])).to(device),
            torch.stack(list(df["phage_embedding"])).to(device),
            torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
        )
    return DataLoader(dataset, batch_size=batch_size)


def dataframe_to_numpy_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a DataFrame with embeddings and interaction types into numpy arrays.

    Bacteria and phage embeddings (and optionally k-mer features) are concatenated into
    a single feature vector per row.

    :param df: DataFrame with columns ``bacterium_embedding``, ``phage_embedding``,
        ``interaction_type``, and optionally ``kmer_embedding``.
    :return: Tuple ``(X, y)`` where X contains the concatenated feature vectors and
        y contains the interaction type labels.
    """
    y = df["interaction_type"]
    if "kmer_embedding" in df.columns:
        cols = ["bacterium_embedding", "phage_embedding", "kmer_embedding"]
    else:
        cols = ["bacterium_embedding", "phage_embedding"]
    X = df[cols]

    arrays = []
    for col in cols:
        col_arr = np.stack(
            [
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                for x in df[col]
            ],
            axis=0,
        )
        arrays.append(col_arr)
    X = np.concatenate(arrays, axis=1)
    return X, y

