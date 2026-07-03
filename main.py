"""
main.py - PBI Pipeline CLI entrypoint.

Usage
-----
Run the full pipeline (embed + train + test):

    python main.py -c model_configs/best_model.yaml

Compute embeddings only (useful for pre-computing in a specific environment
such as the DNABERT2 conda env before running the full pipeline elsewhere):

    python main.py -c model_configs/best_model.yaml --embed-only

Pass a JSON config string directly instead of a file:

    python main.py -j '{"input_perphect": ...}'

See model_configs/example.yaml for all available configuration parameters.
"""

import argparse
import sys

import torch

from pbi_utils.config_parser import parse_config
from pbi_utils.logging import Logging, DEBUG
from pipeline.orchestrator import run_embed_only, run_full_pipeline

Logging.set_logging_level(DEBUG)
logger = Logging()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PBI Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        "-c",
        type=str,
        metavar="PATH",
        help="Path to YAML config file.",
    )
    config_group.add_argument(
        "--json-cli",
        "-j",
        type=str,
        metavar="JSON",
        help="Alternative way to set config. Recieves the entire json as an argument.",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        default=False,
        help=(
            "Only compute and cache embeddings; skip dataset construction, training, and testing. Useful for pre-computing embeddings in environment-specific setups (e.g. DNABERT2 conda env)."
        ),
    )
    return parser


def main() -> None:
    logger.info(f"Running: {' '.join(sys.argv)}")

    parser = _build_parser()
    cli_args = parser.parse_args()

    config = parse_config(cli_args.config, json_cli=cli_args.json_cli)

    if config.torch_num_threads > 0:
        logger.debug(
            f"Setting PyTorch number of threads to {config.torch_num_threads}"
        )
        torch.set_num_threads(config.torch_num_threads)

    if cli_args.embed_only:
        run_embed_only(config)
    else:
        run_full_pipeline(config)


if __name__ == "__main__":
    main()
