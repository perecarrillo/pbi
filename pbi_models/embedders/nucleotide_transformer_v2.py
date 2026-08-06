from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from typing import Literal, Union, get_args
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy
import string
import torch.nn as nn

logger = Logging()


class NT2(AbstractModel):
    MODEL_NAMES = Literal[
        "nucleotide-transformer-2.5b-multi-species",
        "nucleotide-transformer-2.5b-1000g",
        "nucleotide-transformer-500m-human-ref",
        "nucleotide-transformer-500m-1000g",
        "nucleotide-transformer-v2-50m-multi-species",
        "nucleotide-transformer-v2-50m-3mer-multi-species",
        "nucleotide-transformer-v2-100m-multi-species",
        "nucleotide-transformer-v2-500m-multi-species",
        "nucleotide-transformer-v2-250m-multi-species",
    ]
    model_name2short_name = {
        "nucleotide-transformer-2.5b-multi-species": "v1-2.5B-MS",
        "nucleotide-transformer-2.5b-1000g": "v1-2.5B-1KG",
        "nucleotide-transformer-500m-human-ref": "v1-500M-HR",
        "nucleotide-transformer-500m-1000g": "v1-500M-1KG",
        "nucleotide-transformer-v2-50m-multi-species": "50M",
        "nucleotide-transformer-v2-50m-3mer-multi-species": "50M-3mer",
        "nucleotide-transformer-v2-100m-multi-species": "100M",
        "nucleotide-transformer-v2-250m-multi-species": "250M",
        "nucleotide-transformer-v2-500m-multi-species": "500M",
    }

    def __init__(
        self,
        merging_strategy: AbstractMergerStrategy = TruncateStrategy(),
        overlap: int = 0,
        device: str = "cpu",
        model_name: Union[
            MODEL_NAMES, str
        ] = "nucleotide-transformer-v2-50m-multi-species",
        load_model: bool = True,
        batch_size: int = 1,
        comma_separated_gpu_ids: str = "",
    ) -> None:

        self.device = device
        self.overlap = int(overlap)
        self.merging_strategy = merging_strategy
        self.model_name = model_name
        self.batch_size = int(batch_size)

        self.load_model = load_model

        if self.load_model:
            # Not show internal transformers logging messages
            current_log_level = logging.root.level

            if model_name in get_args(self.MODEL_NAMES):
                logger.debug(f"[NT2] Using InstaDeepAI default model weights")
                Logging.set_logging_level()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"InstaDeepAI/{model_name}", trust_remote_code=True
                )
                self.model = AutoModelForMaskedLM.from_pretrained(
                    f"InstaDeepAI/{model_name}", trust_remote_code=True
                )
                Logging.set_logging_level(current_log_level)

            else:
                logger.debug(f"[NT2] Using model weights from {model_name}")
                Logging.set_logging_level()
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_name, trust_remote_code=True
                )
                Logging.set_logging_level(current_log_level)

            if comma_separated_gpu_ids != "":
                logger.debug(
                    f"[NT2] Using DataParallel with GPUs: {comma_separated_gpu_ids}"
                )
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=[int(i) for i in comma_separated_gpu_ids.split(",")],
                )

            self.model.to(self.device)
            self.model.eval()

            self.max_seq_len = (
                self.tokenizer.model_max_length - 2
            ) * 6  # NT tokenizes the sequence as 6-mers, and max_seq_len is for the tokenized sequence. (-1 for the special tokens)

            logger.debug(
                f"[NT2] Max sequence length for Nucleotide Transformer: {self.max_seq_len}"
            )

    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Compute the embeddings
            attention_mask = tokens != self.tokenizer.pad_token_id
            output = self.model(
                tokens,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Compute sequences embeddings
            embeddings = output["hidden_states"][-1]  # [B, L, H]

            # Add embed dimension axis
            attention_mask_unsq = torch.unsqueeze(attention_mask, dim=-1).to(
                embeddings.dtype
            )

            # Compute mean embeddings per sequence
            mean_embed = (embeddings * attention_mask_unsq).sum(
                dim=1
            ) / attention_mask_unsq.sum(dim=1)

            return mean_embed

    def _encode(self, dna_sequence: list[str]) -> torch.Tensor:
        # Tokenize the entire sequence at once
        # tokens_ids = self.tokenizer.batch_encode_plus(dna_sequence, return_tensors="pt", padding=True, truncation=True)["input_ids"].to("cpu") # Keep on CPU for OOM reasons, will be moved to device later
        tokens_ids = self.tokenizer.batch_encode_plus(
            dna_sequence, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        return tokens_ids

    def raw_name(self) -> str:
        short_name = self.model_name2short_name[self.model_name] if self.model_name in self.model_name2short_name else ''.join([x for x in self.model_name if x in (string.ascii_letters + '0123456789')])
        return f"NT2-RAW-{short_name}-ov{self.overlap}-maxlen{self.max_seq_len}"

    def name(self) -> str:
        return f"NT2-{self.merging_strategy.name()}-{self.model_name2short_name[self.model_name] if self.model_name in self.model_name2short_name else ''.join([x for x in self.model_name if x in (string.ascii_letters + '0123456789')])}-ov{self.overlap}"

    def __repr__(self) -> str:
        return f"NT2(merging_strategy={self.merging_strategy}, overlap={self.overlap}, model_name='{self.model_name}')"
