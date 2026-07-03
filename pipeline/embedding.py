"""
Embedding creation function for bacteria and phages.
"""

from typing import List, Literal
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

        # Build a temporary series with one embedding per organism row
        embed_col = f"embedding_{model.name()}"
        ids = df[id_col].tolist()
        sequences = df[seq_col].tolist()

        embeddings = []
        for org_id, sequence in tqdm(
            zip(ids, sequences),
            total=len(ids),
            desc=f"{organism_type.capitalize()} embeddings ({model.name()})",
        ):
            if compute_flag == "auto" and output_manager.has_key(
                id=org_id, model_name=model.name()
            ):
                logger.trace(
                    f"Loading cached embedding for {id_col}={org_id}"
                )
                embedding = output_manager.load_embedding(
                    id=org_id, model_name=model.name(), device=device
                )
            else:
                logger.trace(f"Computing embedding for {id_col}={org_id}")
                embedding = model.embed(sequence)
            embeddings.append(embedding)

        output_manager.save_embeddings_batch(
            ids, embeddings, model_name=model.name(), overwrite=True
        )
