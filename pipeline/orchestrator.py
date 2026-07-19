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

import copy
import json
import os
import time
from typing import List

import numpy as np
import torch
import yaml

from pbi_utils.config_parser import Config
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput
from pbi_utils.logging import Logging

from pipeline.data import make_dataset
from pipeline.dimensionality import fit_pca, reduce_dimensionality, transform_pca
from pipeline.embedding import create_embeddings
from pipeline.evaluation import compute_metrics, test_model, test_nn_ensemble
from pipeline.split import load_predefined_test_set, split_dataset
from pipeline.stats import Stats
from pipeline.threshold import sweep_thresholds
from pipeline.training import kfold_train, kfold_train_ensemble, train_model

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


def _set_seeds(seed: int) -> None:
    """Set global random seeds for reproducibility (torch + numpy)."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _train_ensemble(
    train_df,
    config: Config,
    bacterium_embed_size: int,
    phage_embed_size: int,
    progressbar_prefix: str = "",
) -> List[torch.nn.Module]:
    """
    Train ``config.ensemble_size`` independently-seeded models on *train_df*.

    When ``k_folds_cv <= 1`` each model is trained once on the full *train_df*.
    When ``k_folds_cv > 1`` each fold trains the full ensemble and evaluates it
    as a single unit (averaged predictions), then the ensemble is retrained on
    100% of *train_df* at the end.

    :returns: List of trained ``nn.Module`` instances.
    """
    tc = config.training_config
    ensemble: List[torch.nn.Module] = []

    for i in range(config.ensemble_size):
        member_seed = config.seed + i
        _set_seeds(member_seed)
        model_i = config.classifier(
            bacterium_embed_size,
                phage_embed_size,
                kmer_dim=config.kmer_dim,
                **config.classifier_params,
        )
        if isinstance(model_i, torch.nn.Module):
            model_i.to(config.device)
        desc = f"{progressbar_prefix}Ensemble member {i + 1}/{config.ensemble_size}"
        if tc.k_folds_cv <= 1:
            train_model(
                train_df,
                model_i,
                training_config=tc,
                device=config.device,
                verbose=1,
                progressbar_description=desc,
            )
        ensemble.append(model_i)

    # If k-fold CV is requested with ensemble, delegate to the combined helper
    if tc.k_folds_cv > 1:
        ensemble = kfold_train_ensemble(
            train_df,
            config=config,
            bacterium_embed_size=bacterium_embed_size,
            phage_embed_size=phage_embed_size,
        )

    return ensemble

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
    4. Split into train and test sets (or load a predefined test file).
       For non-predefined splits: PCA is fitted on train only (no leakage).
       For predefined splits: PCA is fitted on the full training set.
    5. Optionally: threshold calibration using a nested calibration split.
    6. Train the ensemble (or single model when ensemble_size=1).
    7. Evaluate on the test set with the configured / calibrated threshold.
    8. Save model checkpoint(s), ensemble metadata, run config, stats TSV,
       and a detailed results JSON.

    :param config: Parsed pipeline configuration.
    """
    logger.info("Running full pipeline.")

    # Set global seeds at pipeline start
    _set_seeds(config.seed)

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
        kmer_config=config.kmer_features,
        bacteria_df=bacteria_df,
        phages_df=phages_df,
    )

    tc = config.training_config

    if tc.do_test:
        if tc.test_split_strategy == "predefined":
            # Predefined path: PCA fitted on full training set
            dataset, pca_bact, pca_phag = reduce_dimensionality(
                dataset,
                tc.reduce_dimensionality,
                config.output_dir,
                tc.n_components_bacteria,
                tc.n_components_phages,
            )
            train = dataset
            test_raw = load_predefined_test_set(
                tc.test_dataset_path,
                bacteria_model_names,
                phages_model_names,
                output_manager,
                config.device,
                kmer_config=config.kmer_features,
                bacteria_df=bacteria_df,
                phages_df=phages_df,
            )
            # Apply the same PCA fitted on training data
            if pca_bact is not None and pca_phag is not None:
                test = transform_pca(test_raw, pca_bact, pca_phag)
            else:
                test = test_raw
        else:
            # Non-predefined path: split first, then fit PCA on train only.
            train_raw, test_raw = split_dataset(
                dataset,
                strategy=tc.test_split_strategy,
                test_size=tc.test_size,
                n_holdout=tc.n_holdout_test,
                random_state=config.seed,
            )
            if tc.reduce_dimensionality != "none":
                pca_bact, pca_phag = fit_pca(
                    train_raw,
                    tc.n_components_bacteria,
                    tc.n_components_phages,
                    output_dir=config.output_dir,
                )
                train = transform_pca(train_raw, pca_bact, pca_phag)
                test = transform_pca(test_raw, pca_bact, pca_phag)
            else:
                pca_bact, pca_phag = None, None
                train = train_raw
                test = test_raw
    else:
        # No test set: fit PCA on the full dataset
        dataset, pca_bact, pca_phag = reduce_dimensionality(
            dataset,
            tc.reduce_dimensionality,
            config.output_dir,
            tc.n_components_bacteria,
            tc.n_components_phages,
        )
        train = dataset
        test = None

    logger.info(f"Train dataset size: {len(train)}")
    if tc.do_test:
        logger.info(f"Test dataset size: {len(test)}")

    bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
    phage_embed_size = len(train["phage_embedding"].iloc[0])

    # Threshold calibration
    calibrated_threshold = None
    threshold = 0.5  # default

    if config.calibrate_threshold and tc.do_test:
        logger.info(
            "calibrate_threshold=True: running nested calibration split to find optimal threshold."
        )
        # Hold out 15% of train for calibration
        from sklearn.model_selection import train_test_split as tts
        train_sub, calib_df = tts(
            train, test_size=0.15, random_state=config.seed, shuffle=True
        )
        logger.info(
            f"Calibration split: {len(train_sub)} train, {len(calib_df)} calibration samples."
        )
        # Train ensemble on 85% subset
        calib_ensemble = _train_ensemble(
            train_sub, config, bacterium_embed_size, phage_embed_size,
            progressbar_prefix="[Calibration] "
        )
        # Get probabilities on calibration set
        from pipeline.evaluation import get_ensemble_probabilities
        calib_probs, calib_labels = get_ensemble_probabilities(
            calib_df, calib_ensemble, tc.batch_size, config.device
        )
        # Find optimal threshold
        best_thr, best_f1 = sweep_thresholds(calib_probs, calib_labels)
        logger.info(
            f"Calibration complete: best threshold={best_thr:.2f}, F1={best_f1:.4f}"
        )
        calibrated_threshold = best_thr
        threshold = best_thr
        # Clean up calibration ensemble memory
        del calib_ensemble

    stats = Stats(config)
    # Training
    t_train = time.perf_counter()

    ensemble = _train_ensemble(
        train, config, bacterium_embed_size, phage_embed_size
    )
    stats.update_classifier(ensemble[0])
    stats.update_ensemble_info(
        ensemble_size=config.ensemble_size,
        seed=config.seed,
        split_strategy=tc.test_split_strategy,
        threshold=threshold,
        calibrated_threshold=calibrated_threshold,
    )

    train_time = time.perf_counter() - t_train

    # Training confusion matrix: evaluate ensemble on training data
    if config.ensemble_size > 1:
        train_cm, _ = test_nn_ensemble(
            train, ensemble, batch_size=tc.batch_size, device=config.device,
            threshold=threshold, silent=True
        )
    else:
        train_cm, _ = test_model(
            train, ensemble[0], batch_size=tc.batch_size, device=config.device, silent=True
        )
    stats.update_train_results(train_cm, train_time)

    # Testing
    cm_test = None
    test_time = None
    if tc.do_test:
        t_test = time.perf_counter()
        if config.ensemble_size > 1:
            cm_test, _ = test_nn_ensemble(
                test, ensemble, batch_size=tc.batch_size, device=config.device,
                threshold=threshold
            )
        else:
            cm_test, _ = test_model(
                test, ensemble[0], batch_size=tc.batch_size, device=config.device
            )
        test_time = time.perf_counter() - t_test
        stats.update_test_results(cm_test, test_time)

    # Save outputs
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

        # Model checkpoint
        if isinstance(ensemble[0], torch.nn.Module):
            if config.ensemble_size == 1:
                model_path = os.path.join(config.output_dir, "trained_model.pth")
                torch.save(ensemble[0].state_dict(), model_path)
                logger.info(f"Trained model saved to: {model_path}")
                model_files = ["trained_model.pth"]
            else:
                model_files = []
                for i, member in enumerate(ensemble):
                    fname = f"trained_model_{i}.pth"
                    model_path = os.path.join(config.output_dir, fname)
                    torch.save(member.state_dict(), model_path)
                    model_files.append(fname)
                logger.info(
                    f"Ensemble of {config.ensemble_size} models saved to: {config.output_dir}"
                )
        else:
            # TODO: SklearnClassifier save method
            logger.warning(
                "Saving sklearn models currently not supported. Not saving the model."
            )
            model_files = []

        # Ensemble metadata JSON
        per_model_seeds = [config.seed + i for i in range(config.ensemble_size)]
        metadata = {
            "ensemble_size": config.ensemble_size,
            "seed": config.seed,
            "per_model_seeds": per_model_seeds,
            "classifier": config.classifier.__name__,
            "classifier_params": config.classifier_params,
            "threshold": threshold,
            "calibrated_threshold": calibrated_threshold,
            "split_strategy": tc.test_split_strategy,
            "model_files": model_files,
        }
        metadata_path = os.path.join(config.output_dir, "ensemble_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Ensemble metadata saved to: {metadata_path}")

        # Detailed results JSON
        results: dict = {
            "ensemble_size": config.ensemble_size,
            "seed": config.seed,
            "split_strategy": tc.test_split_strategy,
            "threshold": threshold,
            "calibrated_threshold": calibrated_threshold,
            "train_metrics": None,
            "test_metrics": None,
        }
        if train_cm is not None:
            tn, fp, fn, tp = (
                int(train_cm[0][0]), int(train_cm[0][1]),
                int(train_cm[1][0]), int(train_cm[1][1])
            )
            acc, rec, f1 = compute_metrics(tn, fp, fn, tp)
            results["train_metrics"] = {
                "accuracy": acc, "recall": rec, "f1": f1,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }
        if cm_test is not None:
            tn, fp, fn, tp = (
                int(cm_test[0][0]), int(cm_test[0][1]),
                int(cm_test[1][0]), int(cm_test[1][1])
            )
            acc, rec, f1 = compute_metrics(tn, fp, fn, tp)
            results["test_metrics"] = {
                "accuracy": acc, "recall": rec, "f1": f1,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }
        results_path = os.path.join(config.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")

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
                "Classifier\tEpochs\tBS\tLR\tEnsemble size\tSplit strategy\tThreshold\t"
                "Train Elapsed time (s)\t"
                "Train True Positive\tTrain False Positive\tTrain False Negative\t"
                "Train True Negative\tTrain Accuracy\tTrain Recall\t"
                "Train F1 Score\tTest Elapsed time (s)\tTest True Positive\t"
                "Test False Positive\tTest False Negative\tTest True Negative\t"
                "Test Accuracy\tTest Recall\tTest F1 Score"
            )
            stats.log(lambda msg: f.write(msg + "\n"))
        logger.info(f"Run stats saved to: {stats_path}")

    logger.info("Pipeline complete.")
