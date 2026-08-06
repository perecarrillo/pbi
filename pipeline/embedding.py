"""
Embedding creation function for bacteria and phages.
"""

from typing import List, Literal
import torch
import pandas as pd
from tqdm import tqdm

from pbi_models.embedders.abstract_model import AbstractModel
from pbi_utils.data_manager import EmbeddingsManager
from pbi_utils.logging import Logging
from pbi_utils.types import CACHED_EMBEDDINGS_OPTION

logger = Logging()

OrganismType = Literal["bacterium", "phage"]


def create_embeddings(
    organism_type: OrganismType,
    models: List[AbstractModel],
    compute_embeddings_flags: List[CACHED_EMBEDDINGS_OPTION],
    df: pd.DataFrame,
    output_manager: EmbeddingsManager,
) -> None:
    """
    Create and cache embeddings for all organisms in *df* using the provided models.

    If ``compute_embeddings_flags[i]`` is ``True``, the embedding is always recomputed.
    If it is ``"auto"``, the embedding is loaded from the cache when available and
    recomputed otherwise.  If it is ``False`` (``use_cached_embeddings=True`` in the
    config), the model is not loaded and this function skips it entirely.

    :param organism_type: ``"bacterium"`` or ``"phage"``.  Determines the column
        names used in *df* (``{organism_type}_id``, ``{organism_type}_sequence``).
    :param models: Instantiated embedding models.  Models not loaded
        (``use_cached_embeddings=True``) are skipped automatically.
    :param compute_embeddings_flags: Per-model flags controlling cache behaviour.
        Matches the ``compute_{organism_type}s_embeddings`` list from the config.
    :param df: DataFrame containing the organisms.  Must have columns
        ``{organism_type}_id`` and ``{organism_type}_sequence``.
    :param output_manager: :class:`EmbeddingsManager` used for saving/loading.
    """
    id_col = f"{organism_type}_id"
    seq_col = f"{organism_type}_sequence"

    loaded_model_names = [
        f"embedding_{model.name()}" for model in models if model.is_loaded()
    ]
    logger.info(
        f"Creating embeddings for {len(loaded_model_names)} {organism_type} models..."
    )

    for model, compute_flag in zip(models, compute_embeddings_flags):
        if not model.is_loaded():
            logger.debug(
                f"Skipping {organism_type} model {model.name()} "
                f"(use_cached_embeddings=True)."
            )
            continue

        device = model.device
        logger.debug(
            f"Creating {organism_type} embeddings for model {model.name()}..."
        )

        merged_model_name = model.name()
        raw_model_name = model.raw_name()
        ids = df[id_col].tolist()
        sequences = df[seq_col].tolist()

        embeddings = []
        for org_id, sequence in tqdm(
            zip(ids, sequences),
            total=len(ids),
            desc=f"{organism_type.capitalize()} embeddings ({model.name()})",
        ):
            # 1. Check if merged embedding already exists
            if compute_flag == "auto" and output_manager.has_key(
                id=org_id, model_name=merged_model_name
            ):
                logger.trace(
                    f"Loading cached merged embedding for {id_col}={org_id}"
                )
                embedding = output_manager.load_embedding(
                    id=org_id, model_name=merged_model_name, device=device
                )
            # 2. Check if raw unmerged chunk embeddings exist
            elif compute_flag == "auto" and output_manager.has_key(
                id=org_id, model_name=raw_model_name
            ):
                logger.trace(
                    f"Loading cached raw chunk embeddings and applying {model.merging_strategy.name()} for {id_col}={org_id}"
                )
                raw_embeds = output_manager.load_embedding(
                    id=org_id, model_name=raw_model_name, device=device, keep_shape=True
                )
                seq_chunks = model._split_sequence(sequence)
                if model.merging_strategy.name() == "TruncateStrategy":
                    seq_chunks_merged = [seq_chunks[0]]
                    raw_embeds_merged = raw_embeds[:1]
                elif model.merging_strategy.name() == "BottomTruncateStrategy":
                    seq_chunks_merged = [seq_chunks[-1]]
                    raw_embeds_merged = raw_embeds[-1:]
                elif model.merging_strategy.name() == "TopBottomTruncateStrategy":
                    seq_chunks_merged = [seq_chunks[0], seq_chunks[-1]]
                    raw_embeds_merged = torch.stack([raw_embeds[0], raw_embeds[-1]], dim=0)
                else:
                    seq_chunks_merged = seq_chunks
                    raw_embeds_merged = raw_embeds

                embedding = model.merging_strategy.merge(seq_chunks_merged, raw_embeds_merged)
                output_manager.save_embedding(
                    id=org_id, embedding=embedding, model_name=merged_model_name, overwrite=True
                )
            # 3. Compute raw chunk embeddings, save raw, merge, and save merged
            else:
                logger.trace(f"Computing embedding for {id_col}={org_id}")
                raw_embeds = model.embed_raw(sequence)
                output_manager.save_embedding(
                    id=org_id, embedding=raw_embeds, model_name=raw_model_name, overwrite=True
                )
                seq_chunks = model._split_sequence(sequence)
                if model.merging_strategy.name() == "TruncateStrategy":
                    seq_chunks_merged = [seq_chunks[0]]
                    raw_embeds_merged = raw_embeds[:1]
                elif model.merging_strategy.name() == "BottomTruncateStrategy":
                    seq_chunks_merged = [seq_chunks[-1]]
                    raw_embeds_merged = raw_embeds[-1:]
                elif model.merging_strategy.name() == "TopBottomTruncateStrategy":
                    seq_chunks_merged = [seq_chunks[0], seq_chunks[-1]]
                    raw_embeds_merged = torch.stack([raw_embeds[0], raw_embeds[-1]], dim=0)
                else:
                    seq_chunks_merged = seq_chunks
                    raw_embeds_merged = raw_embeds

                embedding = model.merging_strategy.merge(seq_chunks_merged, raw_embeds_merged)
                output_manager.save_embedding(
                    id=org_id, embedding=embedding, model_name=merged_model_name, overwrite=True
                )

            embeddings.append(embedding)

        output_manager.save_embeddings_batch(
            ids, embeddings, model_name=merged_model_name, overwrite=True
        )
