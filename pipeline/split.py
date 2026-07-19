"""
Train/test split strategies.

Provides a ``split_dataset()`` function that supports five modes:

* ``"random"``:    random shuffle split
* ``"phage"``:     hold out N random phage IDs for testing
* ``"bacteria"``:  hold out N random bacteria IDs for testing
* ``"organism"``:  hold out N phages AND all their associated bacteria
* ``"predefined"``: handled upstream by the orchestrator via
  :func:`load_predefined_test_set`; passing it to :func:`split_dataset` raises an error.

The predefined-file path is handled by :func:`load_predefined_test_set`, which
reads a separate CSV and assembles a dataset using the embeddings already cached
by the output manager.
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd

from pbi_utils.config_parser import KmerConfig
from pbi_utils.data_manager import EmbeddingsManager
from pbi_utils.logging import Logging
from pipeline.data import make_dataset

logger = Logging()


def split_dataset(
    dataset: pd.DataFrame,
    strategy: Literal["random", "phage", "bacteria", "organism"],
    test_size: float = 0.2,
    n_holdout: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *dataset* into train and test sets according to *strategy*.

    :param dataset: Full dataset DataFrame with columns ``bacterium_embedding``,
        ``phage_embedding``, ``interaction_type``, ``phage_id``, ``bacterium_id``.
    :param strategy: Split strategy.
        - ``"random"``:   random shuffle split (uses *test_size*).
        - ``"phage"``:    hold out *n_holdout* random phage IDs (uses *n_holdout*).
        - ``"bacteria"``: hold out *n_holdout* random bacteria IDs (uses *n_holdout*).
        - ``"organism"``: hold out *n_holdout* phages AND all bacteria associated with
          them (no organism appears in both train and test).
    :param test_size: Fraction of data for the test set (``strategy="random"`` only).
    :param n_holdout: Number of unique organism IDs to hold out
        (``strategy="phage"``, ``"bacteria"``, or ``"organism"`` only).
    :param random_state: Random seed for reproducibility.
    :return: ``(train_df, test_df)`` DataFrames.
    :raises ValueError: For unsupported strategies or missing parameters.
    """
    if strategy == "random":
        from sklearn.model_selection import train_test_split

        return train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    elif strategy == "phage":
        if n_holdout is None:
            raise ValueError(
                "n_holdout_test must be set in the config when test_split_strategy='phage'."
            )
        test_ids = (
            dataset["phage_id"]
            .drop_duplicates()
            .sample(n=n_holdout, random_state=random_state)
            .tolist()
        )
        test_df = dataset[dataset["phage_id"].isin(test_ids)].reset_index(drop=True)
        train_df = dataset[~dataset["phage_id"].isin(test_ids)].reset_index(drop=True)
        logger.info(
            f"Phage-based split: held out {len(test_ids)} phage IDs for testing "
            f"({len(test_df)} test rows, {len(train_df)} train rows)."
        )
        return train_df, test_df

    elif strategy == "bacteria":
        if n_holdout is None:
            raise ValueError(
                "n_holdout_test must be set in the config when "
                "test_split_strategy='bacteria'."
            )
        test_ids = (
            dataset["bacterium_id"]
            .drop_duplicates()
            .sample(n=n_holdout, random_state=random_state)
            .tolist()
        )
        test_df = dataset[dataset["bacterium_id"].isin(test_ids)].reset_index(drop=True)
        train_df = dataset[~dataset["bacterium_id"].isin(test_ids)].reset_index(drop=True)
        logger.info(
            f"Bacteria-based split: held out {len(test_ids)} bacteria IDs for testing "
            f"({len(test_df)} test rows, {len(train_df)} train rows)."
        )
        return train_df, test_df

    elif strategy == "organism":
        if n_holdout is None:
            raise ValueError(
                "n_holdout_test must be set in the config when "
                "test_split_strategy='organism'."
            )
        held_out_phage_ids = (
            dataset["phage_id"]
            .drop_duplicates()
            .sample(n=n_holdout, random_state=random_state)
            .tolist()
        )
        held_out_bact_ids = (
            dataset[dataset["phage_id"].isin(held_out_phage_ids)]["bacterium_id"]
            .drop_duplicates()
            .tolist()
        )
        in_test = dataset["phage_id"].isin(held_out_phage_ids) | dataset[
            "bacterium_id"
        ].isin(held_out_bact_ids)
        test_df = dataset[in_test].reset_index(drop=True)
        train_df = dataset[~in_test].reset_index(drop=True)

        test_fraction = len(test_df) / len(dataset)
        logger.info(
            f"Organism-based split: held out {len(held_out_phage_ids)} phages and "
            f"{len(held_out_bact_ids)} associated bacteria for testing "
            f"({len(test_df)} test rows [{test_fraction:.1%}], {len(train_df)} train rows)."
        )
        if test_fraction > 0.5:
            logger.warning(
                f"Organism split produced a test set larger than 50% of the data "
                f"({test_fraction:.1%}). Consider reducing n_holdout_test."
            )

        # sanity check
        train_phages = set(train_df["phage_id"].unique())
        train_bact = set(train_df["bacterium_id"].unique())
        test_phages = set(test_df["phage_id"].unique())
        test_bact = set(test_df["bacterium_id"].unique())
        assert not train_phages & test_phages, "Organism split: phage ID leakage detected!"
        assert not train_bact & test_bact, "Organism split: bacterium ID leakage detected!"

        return train_df, test_df

    elif strategy == "predefined":
        raise ValueError(
            "test_split_strategy='predefined' is handled by the orchestrator directly "
            "via load_predefined_test_set(). Do not call split_dataset() with this strategy."
        )

    else:
        raise ValueError(
            f"Unknown test_split_strategy {strategy!r}. "
            "Allowed values: 'random', 'phage', 'bacteria', 'organism', 'predefined'."
        )


def load_predefined_test_set(
    test_dataset_path: str,
    bacteria_model_names: List[str],
    phages_model_names: List[str],
    output_manager: EmbeddingsManager,
    device: str,
    kmer_config: KmerConfig | None = None,
    bacteria_df: pd.DataFrame | None = None,
    phages_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load a predefined test set from a separate CSV file.

    The CSV must have columns ``bacterium_id``, ``phage_id``, and
    ``interaction_type``.  Embeddings are loaded from *output_manager* using
    the same models as the training set.

    :param test_dataset_path: Path to the test CSV file.
    :param bacteria_model_names: Bacteria embedding model names.
    :param phages_model_names: Phage embedding model names.
    :param output_manager: EmbeddingsManager for loading cached embeddings.
    :param device: Device to load tensors onto.
    :param kmer_config: Optional KmerConfig specifying k values and sequence columns.
    :param bacteria_df: Optional DataFrame with bacteria sequences.
    :param phages_df: Optional DataFrame with phage sequences.
    :return: Test dataset DataFrame with ``bacterium_embedding`` and
        ``phage_embedding`` columns (plus ``kmer_embedding`` if configured).
    """
    logger.info(f"Loading predefined test set from: {test_dataset_path}")
    test_couples_df = pd.read_csv(test_dataset_path)
    return make_dataset(
        test_couples_df,
        bacteria_model_names,
        phages_model_names,
        output_manager,
        device,
        kmer_config=kmer_config,
        bacteria_df=bacteria_df,
        phages_df=phages_df,
    )
