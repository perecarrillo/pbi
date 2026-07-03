"""
Top-level pipeline orchestration.

Provides two entry points:

* :func:`run_embed_only`: compute and cache embeddings without training.
* :func:`run_full_pipeline`: full end-to-end run: embed -> dataset -> reduce
  -> split -> train -> test -> save.

Both functions accept a parsed :class:`~pbi_utils.config_parser.Config` object
and delegate to the individual pipeline modules for each step.
"""

from __future__ import annotations

import os
import time

import torch
import yaml

from pbi_utils.config_parser import Config
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput
from pbi_utils.logging import Logging

from pipeline.data import make_dataset
from pipeline.dimensionality import reduce_dimensionality, transform_pca
from pipeline.embedding import create_embeddings
from pipeline.evaluation import test_model
from pipeline.split import split_dataset, load_predefined_test_set
from pipeline.stats import Stats
from pipeline.training import train_model, kfold_train

logger = Logging()


### Shared setup helpers

def _setup(config: Config):
    """Load input data and create the embeddings output manager."""
    bacteria_df, phages_df, couples_df = PerphectDataInput(
        input_paths=config.input_perphect
    ).load()

    output_manager = H5pyEmbeddingsManager(config.embeddings_dir)
    return bacteria_df, phages_df, couples_df, output_manager


def _run_embeddings(config: Config, bacteria_df, phages_df, output_manager):
    """Compute and cache embeddings for all configured models."""
    create_embeddings(
        organism_type="bacterium",
        models=config.bacteria_embedding_models,
        compute_embeddings_flags=config.compute_bacteria_embeddings,
        df=bacteria_df,
        output_manager=output_manager,
    )
    create_embeddings(
        organism_type="phage",
        models=config.phages_embedding_models,
        compute_embeddings_flags=config.compute_phages_embeddings,
        df=phages_df,
        output_manager=output_manager,
    )


### Public entry points

def run_embed_only(config: Config) -> None:
    """
    Compute and cache embeddings only - skip dataset construction, training, and testing.

    This is useful for pre-computing embeddings in a specific environment (e.g.
    the DNABERT2 conda environment) before running the full pipeline in another.

    :param config: Parsed pipeline configuration.
    """
    logger.info("Running in embed-only mode.")

    bacteria_df, phages_df, _, output_manager = _setup(config)
    _run_embeddings(config, bacteria_df, phages_df, output_manager)

    logger.info("Embedding creation complete.")


def run_full_pipeline(config: Config) -> None:
    """
    Run the complete PBI pipeline end-to-end.

    Steps:
    1. Load input CSV files.
    2. Compute / load cached embeddings for all configured models.
    3. Build the training dataset by merging and concatenating embeddings.
    4. Apply dimensionality reduction (PCA if configured).
    5. Split into train and test sets (or load a predefined test file).
    6. Instantiate the classifier.
    7. Train (single run or k-fold CV).
    8. Evaluate on the test set.
    9. Save model checkpoint, run configuration, and stats TSV.

    :param config: Parsed pipeline configuration.
    """
    logger.info("Running full pipeline.")

    # Setup
    bacteria_df, phages_df, couples_df, output_manager = _setup(config)

    # Embeddings
    _run_embeddings(config, bacteria_df, phages_df, output_manager)

    if not config.training_config.do_train:
        logger.info("do_train=False, skipping training and testing.")
        return

    # Dataset construction
    bacteria_model_names = [m.name() for m in config.bacteria_embedding_models]
    phages_model_names = [m.name() for m in config.phages_embedding_models]

    dataset = make_dataset(
        couples_df,
        bacteria_model_names,
        phages_model_names,
        output_manager,
        config.device,
    )

    # Dimensionality reduction (fit on the full couples dataset)
    dataset, pca_bact, pca_phag = reduce_dimensionality(
        dataset,
        config.training_config.reduce_dimensionality,
        config.output_dir,
        config.training_config.n_components_bacteria,
        config.training_config.n_components_phages,
    )

    # Train / test split
    tc = config.training_config

    if tc.do_test:
        if tc.test_dataset_path is not None:
            # Load a predefined test CSV (e.g. predphi_test_dataset.csv)
            train = dataset
            test_raw = load_predefined_test_set(
                tc.test_dataset_path,
                bacteria_model_names,
                phages_model_names,
                output_manager,
                config.device,
            )
            # Apply the same dimensionality reduction fitted on training data
            if pca_bact is not None and pca_phag is not None:
                test = transform_pca(test_raw, pca_bact, pca_phag)
            else:
                test = test_raw
        else:
            train, test = split_dataset(
                dataset,
                strategy=tc.test_split_strategy,
                test_size=tc.test_size,
                n_holdout=tc.n_holdout_test,
            )
    else:
        train = dataset

    logger.info(f"Train dataset size: {len(train)}")
    if tc.do_test:
        logger.info(f"Test dataset size: {len(test)}")

    # Classifier creation
    bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
    phage_embed_size = len(train["phage_embedding"].iloc[0])
    model = config.classifier(
        bacterium_embed_size, phage_embed_size, **config.classifier_params
    )

    stats = Stats(config)
    stats.update_classifier(model)

    # Training
    t_train = time.perf_counter()

    if tc.k_folds_cv <= 1:
        cm = train_model(
            train,
            model,
            training_config=tc,
            device=config.device,
            verbose=1,
            progressbar_description="Training...",
        )
    else:
        cm = kfold_train(
            train,
            model,
            training_config=tc,
            device=config.device,
        )

    train_time = time.perf_counter() - t_train
    stats.update_train_results(cm, train_time)

    # Testing
    if tc.do_test:
        t_test = time.perf_counter()
        cm_test, _ = test_model(
            test, model, batch_size=tc.batch_size, device=config.device
        )
        test_time = time.perf_counter() - t_test
        stats.update_test_results(cm_test, test_time)

    # Save outputs
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

        # Model checkpoint
        model_path = os.path.join(config.output_dir, "trained_model.pth")
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
            logger.info(f"Trained model saved to: {model_path}")
        else:
            # TODO: SklearnClassifier save method
            logger.warning(
                "Saving sklearn models currently not supported. Not saving the model."
            )

        # Run configuration
        training_config_path = os.path.join(config.output_dir, "training_config.yaml")
        with open(training_config_path, "w") as f:
            yaml.dump(config.raw_dict, f)
        logger.info(f"Training config saved to: {training_config_path}")

        # Stats TSV
        stats_path = os.path.join(config.output_dir, "stats.tsv")
        with open(stats_path, "w") as f:
            f.write(
                "Date\tCommand\tDescription\tBacteria embedder\tPhages embedder\t"
                "Classifier\tEpochs\tBS\tLR\tTrain Elapsed time (s)\t"
                "Train True Positive\tTrain False Positive\tTrain False Negative\t"
                "Train True Negative\tTrain Accuracy\tTrain Weighted accuracy\t"
                "Train F1 Score\tTest Elapsed time (s)\tTest True Positive\t"
                "Test False Positive\tTest False Negative\tTest True Negative\t"
                "Test Accuracy\tTest Weighted accuracy\tTest F1 Score"
            )
            stats.log(lambda msg: f.write(msg + "\n"))
        logger.info(f"Run stats saved to: {stats_path}")

    logger.info("Pipeline complete.")
