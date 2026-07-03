# PBI

## Table of Contents
1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Environment](#environment)
    - [Base Environment](#base-environment)
    - [DNABERT2 Environment](#dnabert2-environment)
    - [Finetuning Nucleotide Transformer v2](#finetuning-nucleotide-transformer-v2)
4. [Execution](#execution)
5. [Available Models](#available-models)
    - [Embedding Models](#embedding-models)
    - [Classifier Models](#classifier-models)
    - [Adding New Models](#adding-new-models)
7. [Available Merging Strategies](#available-merging-strategies)
    - [Adding New Merging Strategies](#adding-new-merging-strategies)
8. [Utilities](#utilities)

## Overview

This repository contains all the code for the new iteration at solving the PBI problem by the CI4CB laboratory.

It includes a framework to test and verify multiple approaches at solving the problem, with a modular architecture where any part can be tuned.

![Framework architecture](doc/images/Architecture.png)

The framework consists in two branches that embed and compress the DNA sequences of the bacterium and phage, and a classifier that makes the prediction. For each of the branches, first, 3 foundation models are used to compute the embeddings of the sequence, by dividing it into multiple subsequences of maximum size (that the model allows) and embedding all the parts. Then, a strategy is used to transform this set of embeddings to one single meta-embedding, to which PCA is applied to reduce even more the dimensionality. Finally, the meta-embeddings from the two branches are given to the classifier and it predicts wether the two organisms have an interaction or not.

Every part of the architecture is configurable, by passing a YAML configuration to the execution. As an example, [model_configs/example.yaml](model_configs/example.yaml) contains all the parameters that can be used with an explanation. In addition, the [Available Models](#available-models) contains all the implemented models that can be used, both for the embedders and classifiers.

## Repository structure

```shell
pbi/
├── README.md # This file.
├── analysis/ # Jupyter notebooks to perform different analysis. See the <Utilities> section.
├── data/ # This folder is not included in the repository. You should create it and add all the data inside.
├── doc/ # Extra documentation for developers, showing the specific details for all the code.
├── finetune_nt2.py # Finetuning script for the Nucleotide Transformer v2. See `run_finetune_nt2.sh` file for an example, or run `python finetune_nt2.py --help` to see all the possible parameters.
├── main.py # Main entrypoint for the framework. Parses arguments and delegates to pipeline/.
├── model_configs/ # YAML configuration files provided as an example. They are used to define all the parameters for each run.
│   ├── example.yaml # Example config with details on all the possible parameters.
│   ├── base.yaml # Basic execution config to use as test.
│   ├── best_model_pbip_datasets.yaml # Configuration of the best model found during the project for the PredPHI data.
│   └── best_model.yaml # Best configuration found during the project for the CI4CB data.
├── pipeline/ # Modular pipeline components. Each module handles one stage of the pipeline.
│   ├── __init__.py # Re-exports run_full_pipeline and run_embed_only.
│   ├── embedding.py # Creates the embeddings for bacteria and phages.
│   ├── data.py # Dataset construction: make_dataset, dataframe_to_tf_dataloader, etc.
│   ├── dimensionality.py # PCA: fit_pca, transform_pca, reduce_dimensionality.
│   ├── training.py # train_model, train_nn_model, kfold_train, LR scheduler factory.
│   ├── evaluation.py # test_model, compute_metrics.
│   ├── split.py # Configurable train/test split strategies.
│   ├── stats.py # Stats class for logging run results to TSV.
│   └── orchestrator.py # Top-level run_full_pipeline() and run_embed_only().
├── pbi_models/ # Implementation of the different classifiers and embedding models.
│   ├── classifiers/ # Classifiers implementation.
│   │   ├── abstract_classifier.py # Abstract classifier class. All the others should inherit from this class, to provide a stable API.
│   │   ├── base.py # Dummy basic classifier, based on an MLP with 1 hidden layer.
│   │   ├── CNN.py # CNN classifiers implementation. Provides CNNClassifier and BasicCNNClassifier.
│   │   ├── MLP.py # MLP classifiers implementation. Provides MLPClassifier and BasicMLPClassifier.
│   │   ├── linear.py # Linear classifier. Provides LinearClassifier.
│   │   └── sklearn_classifier. # Sci-kit learn classifiers. Can use any sklearn classifiers with the addition of LightGBM and XGBoost.
│   └── embedders/ # Embedding models implementation.
│       ├── abstract_model.py # Abstract embedder class. All the others should inherit from this class, to provide a stable API.
│       ├── dnabert2.py # DNABERT2 model implementation. Provides the DNABERT2 class.
│       ├── evo.py # EVO1 model implementation. It is not recommended to use, as it can only deal with sequences of up to 512bp. Use at your own risk. Provides the EVO class.
│       ├── megaDNA.py # MegaDNA model implementation. Provides the MegaDNA class.
│       └── nucleotide_transformer_v2.py # Nucleotide Transformer v2 model implementation. Provides the NT2 class.
├── pbi_utils/ # Additional utilities implementation for the project.
│   ├── embeddings_merging_strategies/ # Implementation of the different merging strategies.
│   │   ├── abstract_merger_strategy.py # Abstract merging class. All the others should inherit from this one to provide a stable API.
│   │   ├── average_strategy.py # Provides the AverageStrategy class.
│   │   ├── max_strategy.py # Provides the MaxStrategy class.
│   │   ├── tfidf_strategy.py # Provides the TfidfStrategy and the Tf4idfStrategy classes.
│   │   ├── tkpert_strategy.py # Provides the TKPertStrategy class.
│   │   └── truncate_strategy.py # Provides the TruncateStrategy, BottomTruncateStrategy and TopBottomTruncateStrategy.
│   ├── config_parser.py # Responsible of dealing with the input. Defines all the input parameters and reads them.
│   ├── data_manager.py # Responsible of dealing with reading and storing the DNA data and embeddings.
│   ├── logging.py # Logging system implementation used for the project.
│   ├── types.py # Defines some types that are used in multiple files.
│   └── utils.py # Provides extra utility functions (e.g. clean_gpu).
├── requirements.txt # Base requirements file.
├── requirements_nt2_finetuning.txt # Requirements file for finetuning the Nucleotide Transformer v2.
├── run.sh # Basic execution example.
├── run_finetune_nt2.sh # Finetuning example.
└── run_gridsearch.sh # Script to perform gridsearch over some user-defined environment variables.
```

## Environment

Each of the foundation models requires an specific version of libraries, so it is much recommended to set up a Python package manager such as conda or micromamba. In this guide, micromamba with an alias to `mm` is used. If using conda instead, just replace `mm` with `conda`.

### Base Environment
This environment will be used to run everything that does not require a specific environment, for now, everything except the DNABERT2 model and finetuning the Nucleotide Transformer v2.

Start by creating a new environment and activating it.
```bash
mm create -n pbi
mm activate -n pbi
```
> [!IMPORTANT]
> **Make sure that you have activated correctly the environment. If not, the next commands will also work but install everything in your base environment, which might cause problems later on.**

Next, install Python 3.10.18 (higher versions might also work, but this is the one used to develop the project).
```bash
mm install "python==3.10.18"
```

To continue, this project has been developed with CUDA version 12.4. Again, it might work with higher versions, but it has not been tested. To install CUDA 12.4, run:
```bash
mm install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti nccl -c nvidia/label/cuda-12.4.0
```

Finally, to install all the pip dependencies, run:
```bash
pip install -r requirements.txt
```

If everything worked correctly, congratulations, you can now start executing things.

### DNABERT2 Environment

> [!TIP]
> If you are not using the DNABERT2 model or are using cached embeddings, do not use this environment, and use the base one instead.

To use DNABERT2, first, you need to download their repository.
```bash
git clone https://huggingface.co/zhihan1996/DNABERT-2-117M
```

Next, enter their folder, and in the file `flash_attn_triton.py` replace all the occurrences of `tl.dot(q, k, trans_b=True)` (or similar, all the ones that have `trans_a` or `trans_b` as a parameters) with the updated syntax of `tl.dot(q, tl.trans(k))`.
> [!IMPORTANT]
> Take your time replacing them, as in some of them it is the first parameter the one that needs to be transposed. If you make a mistake it might continue to work (or it might not) but obtain bad results.

Finally, starting from the last step in the base environment (create a new one called `pbi-dnabert` and follow the same steps), install now the latest version of triton (tested with triton==3.5.1), with:
```bash
pip install --upgrade triton
```
Pip will complain, as the `torch` version is too old to use the last version of `triton`. Just ignore it and it will work anyways.

### Finetuning Nucleotide Transformer v2

> [!TIP]
> If you are not using a finetuned model or are using cached embeddings, do not use this environment, and use the base one instead.

To finetune the Nucleotide Transformer v2 model, create a new environment called `pbi-finetune` and follow the same steps as in the base environment.

Finally, install the extra dependencies for finetuning:
```bash
pip install -r requirements_nt2_finetuning.txt
```

If you want to use then this finetuned model, you will need to run the main framework in this environment. 

## Execution

Once you have your environment correctly set up, the data prepared, and a YAML configuration file created, for example [model_configs/base.yaml](model_configs/base.yaml), to run the framework, execute:
```bash
python main.py -c model_configs/base.yaml
```

This will compute the embeddings for all the sequences (or use the cached ones if you already did it), train the classifier that you specified, and test it on the dataset test split. The train and test results metrics will be shown in the terminal.

> [!NOTE]
> If you are computing the embeddings from scratch with a merging strategy different than [*TruncateStrategy*, *BottomTruncateStrategy* or *TopBottomTruncateStrategy*], it will take multiple hours to finish.

### Pre-computing embeddings only

If you are using multiple environments (e.g. DNABERT2 requires its own conda environment), you can compute only the embeddings without running training:

```bash
python main.py -c model_configs/base.yaml --embed-only
```

This is especially useful for pre-computing expensive embeddings in the correct environment (e.g. `pbi-dnabert`) and then running the full training in the base environment where all models have `use_cached_embeddings: true`.

If you are using DNABERT2 as embedding model, make sure to run the execution in the `pbi-dnabert` environment. If you are using a finetuned Nucleotide Transformer v2 model, run it in the `pbi-finetune` environment.
> [!NOTE]
> The DNABERT2 model has high GPU VRAM needs, and must be run on higher-end GPUs, such as NVIDIA A40.

We strongly recommend computing first all the embeddings for all the models that you want to use separatedly, by executing multiple times the framework with only one embedding model at a time (and with the correct environment activated), as they will be cached and the future training will be much smoother.
> [!NOTE]
> The embeddings are cached for each combination of model + merging strategy, so if you change any of them, they will need to be recomputed.

### Configuring the test split

The `training_config.test_split_strategy` option controls how the test set is created:

| Strategy | Description | Required extra field |
|---|---|---|
| `random` (default) | Random shuffle split | `test_size` (fraction, default 0.2) |
| `phage` | Hold out N random phage IDs | `n_holdout_test` (integer) |
| `bacteria` | Hold out N random bacteria IDs | `n_holdout_test` (integer) |

Alternatively, set `test_dataset_path` to a separate CSV file (e.g. a predefined benchmark test split). When set, the split strategy is ignored and PCA (if enabled) is correctly fitted on the training set only and then applied to the test file.

### Configuring the LR scheduler

Set `training_config.lr_schedule` to one of:
- `plateau` (default): `ReduceLROnPlateau`, requires a validation set (`k_folds_cv >= 2`).
- `cosine`: `CosineAnnealingLR`, decays from `learning_rate` to `1e-6` over `epochs` steps. Works without a validation set (`k_folds_cv: 1`).
- `none`: No LR scheduling.

To run the best model found during the project, assuming that all the required embeddings have already been computed, just execute:
```bash
python main.py -c model_configs/best_model.yaml
```

> [!TIP]
> To test a trained model on a custom dataset, you can use the `analysis/model_testing.ipynb` jupyter notebook.

## Available Models
The following models are implemented and can be used in the framework.

### Embedding Models
- Nucleotide Transformer v2: [nature.com/articles/s41592-024-02523-z](https://www.nature.com/articles/s41592-024-02523-z)

- DNABERT2: [arxiv.org/pdf/2306.15006](https://arxiv.org/pdf/2306.15006).

- MegaDNA: [nature.com/articles/s41467-024-53759-4](https://www.nature.com/articles/s41467-024-53759-4)

### Foundation Models Features Comparison

| Model | Base Architecture | Tokenization | Positional Information | Context Length | Parameters | Training Data |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Nucleotide Transformer v2` | Transformer (BERT) | 6-mers | Rotary embeddings | 12K bp | 50M - 500M | <sup>1</sup> |
| `DNABERT2` | Transformer (BERT) | BPE | ALiBi | 10K bp | 117M | <sup>2</sup> |
| `MegaDNA` | Transformer (GPT) | single-nucleotide | None | 96K bp | 145M | <sup>4</sup> |
| `EVO2` | Convolutions (StripedHyena) | single-nucleotide | Rotary embeddings | 131K bp | 1B - 70B | <sup>3</sup> |

**Notes on Training Data:**
* **1:** Human, mammalian, fungi and bacteria genomes
* **2:** Human, mammalian, fungi, invertebrate and bacteria genomes
* **3:** DNA and RNA sequences from all domain of life, including virus
* **4:** Only bacteriophage genomes

EVO2 model is not implemented because it can only run on higher-end GPUs due to its memory requirements, mainly NVIDIA H100 or H200. If you want to use it, you can implement it by following the structure of the other models, inheriting from the `AbstractEmbedder` class.

### Classifier Models
- BasicClassifier: Simple MLP with 1 hidden layer. Has only one hyperparameter, the hidden layer size.
- BasicMLPClassifier: More complex MLP with multiple hidden layers, dropout and batch normalization. Has two hyperparameters, *mlp_params* (a list of integers with the size of each hidden layer) and *dropout* (the dropout probability, equal for all the layers).
- MLPClassifier: Two-branch MLP classifier, with different parameters for each branch. Each branch has its own *<bacterium|phage>_mlp_sizes* parameter, and *dense_dim* (size of the final dense layer before the output) and *dropout* (dropout probability, equal for all the layers) are shared.
- CNNClassifier: Two-branch CNN classifier, with different parameters for each branch. Each branch has its own *<bacterium|phage>_conv_params*, which is a list of tuples with the parameters for each convolutional layer (out_channels, kernel_size, stride). The rest of the parameters are shared: *dense_dim* (size of the final dense layer before the output) and *dropout* (dropout probability, equal for all the layers).
- BasicCNNClassifier: Similar to CNNClassifier, but with a single branch CNN architecture. The parameters are: *cnn_params* (a list of tuples with the parameters for each convolutional layer (out_channels, kernel_size, stride)), *dense_dim* (size of the final dense layer before the output) and *dropout* (dropout probability, equal for all the layers).
- LinearClassifier: Simple linear classifier. No hyperparameters.
- SklearnClassifier: Any Sci-kit learn classifier can be used here. Just specify the *sklearn_model* parameter with the name of the model (for example, `RandomForestClassifier`, `LogisticRegression`, etc.) and the *sklearn_params* parameter with a dictionary of the hyperparameters for that model. In addition, LightGBM and XGBoost classifiers are also supported, by specifying `LGBMClassifier` and `XGBClassifier` respectively.

### Adding New Models

To add a new embedding model or classifier, you need to create a new file inside the corresponding folder in `pbi_models/`, create a new class that inherits from the abstract class (`AbstractEmbedder` or `AbstractClassifier`) and implement all the required methods (the ones tagged as `@abstractmethod` in the abstract class file). The system will automatically detect the new model and will be able to use it, just by specifying its name in the YAML configuration file. You can check the existing models as an example of how to implement it.

> [!NOTE]
> Make sure to follow the input and output specifications of the abstract classes, otherwise the framework will not work correctly.

## Available Merging Strategies
The following merging strategies are implemented and can be used in the framework. See [this paper](https://arxiv.org/pdf/2304.14796) for more details about any of them, or the docstrings inside each class.

- AverageStrategy: Computes the average of all the embeddings.
- MaxStrategy: Computes the maximum value for each dimension across all the embeddings.
- TfidfStrategy: Computes a weighted average of the embeddings, where the weights are computed using the TF-IDF algorithm.
- Tf4idfStrategy: Similar to TfidfStrategy, but uses the TF4-IDF formulas to compute the weights.
- TKPertStrategy: Uses the TKPert algorithm to compute the weights for each embedding, and then computes a weighted average. Has three hyperparameters: *J* (the number of windows to use), *gamma* (the gamma parameter for the PERT function) and *merging_strategy* (the merging strategy used to combine the weighted embeddings, can be `avg` or `concat`). 
- TruncateStrategy: Only considers the first embedding of all the ones embedded by the foundation model.
- BottomTruncateStrategy: Only considers the last embedding of all the ones embedded by the foundation model.
- TopBottomTruncateStrategy: Considers both the first and last embeddings of all the ones embedded by the foundation model, and concatenates them.

### Adding New Merging Strategies
To add a new merging strategy, you need to create a new file inside the `pbi_utils/embeddings_merging_strategies/` folder, create a new class that inherits from the `AbstractMergerStrategy` class and implement at least the `merge()` method. The system will automatically detect the new merging strategy and will be able to use it, just by specifying its name in the YAML configuration file. You can check the existing merging strategies as an example of how to implement it.

> [!NOTE]
> Make sure to follow the input and output specifications of the abstract class, otherwise the framework will not work correctly.

## Utilities

Some bash scripts are also provided to help with specific needs.
- [run.sh](run.sh): Shows an example run test.
- [run_gridsearch.sh](run_gridsearch.sh): Performs a gridsearch over different parameters that can be customized at the start of the file. The intended use is to use environment variables inside the config file (by using `$<env_var>` as a value), and change them inside the script.
- [run_finetune_nt2.sh](run_finetune_nt2.sh): Shows an example of finetuning the Nucleotide Transformer v2.

Additionally, some Jupyter notebooks are also provided, that perform different analysis on the data and models. They are available inside the `analysis/` folder.
- [data_analysis.ipynb](analysis/data_analysis.ipynb): Explores the public and private datasets, and creates one that joins them. It also plots the embeddings from DNABERT2 in a 2D grid to visualize them.
- [embeddings_merging_strategies_analysis.ipynb](analysis/embeddings_merging_strategies_analysis.ipynb): Analyzes the results from the gridsearch over merging strategies, and creates plots to compare them.
- [model_testing.ipynb](analysis/model_testing.ipynb): Explores the results from a trained model, testing it with different datasets and analyzing the predictions.
- [prepare_data_finetuning.ipynb](analysis/prepare_data_finetuning.ipynb): Prepares the data to finetune the Nucleotide Transformer v2 model.