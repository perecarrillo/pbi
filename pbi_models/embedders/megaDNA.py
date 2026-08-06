from typing import Literal
from pbi_models.embedders.abstract_model import AbstractModel
import torch
import os
import urllib.request
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy
from pbi_utils.logging import Logging
from functools import reduce

logger = Logging()


class MegaDNA(AbstractModel):
    def __init__(
        self,
        weights_path: str,
        merging_strategy: AbstractMergerStrategy = TruncateStrategy(),
        overlap: int = 0,
        device: str = "cpu",
        get_layer: Literal["concat", "last"] = "last",
        load_model: bool = True,
    ) -> None:

        self.load_model = load_model
        self.batch_size = 1

        if self.load_model:
            if not os.path.isfile(weights_path):
                logger.warning(
                    f"[MegaDNA] Weights not found, downloading them again and saving them to: {weights_path}"
                )
                os.makedirs(weights_path.rsplit("/", maxsplit=1)[0], exist_ok=True)
                urllib.request.urlretrieve(
                    "https://huggingface.co/lingxusb/megaDNA_updated/resolve/main/megaDNA_phage_145M.pt",
                    weights_path,
                )

            self.device = device
            self.model = torch.load(weights_path, map_location=torch.device(device))
            self.model.eval()
            self.max_seq_len = reduce(lambda x, y: x * y, self.model.max_seq_len) - 1
            logger.debug(
                f"[MegaDNA] Max sequence length for megaDNA: {self.max_seq_len}"
            )

        self.merging_strategy = merging_strategy
        self.overlap = int(overlap)

        self.get_layer = get_layer

    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(tokens, return_value="embedding")

        if self.get_layer == "last":
            last_embed = output[-1]
            mean_embed = last_embed.mean(dim=(0, 1)).unsqueeze(
                0
            )  # torch.Size([1, 196])
        else:  # concat
            output = [o.mean(dim=(0, 1), keepdim=True) for o in output]
            mean_embed = torch.cat(output, dim=2).squeeze(
                0
            )  # torch.Size([1, 196 + 256 + 512])

        return mean_embed

    def __vocabulary(self, char: str) -> int:
        match char:
            case "**":
                return 0
            case "A":
                return 1
            case "T":
                return 2
            case "C":
                return 3
            case "G":
                return 4
            case "#":
                return 5
            case _:
                return 0

    def _encode(self, dna_sequences: list[str]) -> torch.Tensor:
        # Encode a batch of sequences
        encoded_sequences = [self._encode_single(seq) for seq in dna_sequences]
        return torch.cat(encoded_sequences, dim=0)

    def _encode_single(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(
                f"[MegaDNA] Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), trimming it"
            )
            dna_sequence = dna_sequence[: self.max_seq_len]
        elif len(dna_sequence) < self.max_seq_len:
            old_seq = dna_sequence
            dna_sequence = [0] * self.max_seq_len
            dna_sequence[: len(old_seq)] = old_seq  # type: ignore

        return torch.tensor(
            [self.__vocabulary(nt) for nt in dna_sequence] + [self.__vocabulary("#")],
        ).unsqueeze(dim=0)

    def raw_name(self) -> str:
        return f"MegaDNA-RAW-{self.get_layer}-ov{self.overlap}-maxlen{self.max_seq_len}"

    def name(self) -> str:
        return (
            f"MegaDNA-{self.merging_strategy.name()}-{self.get_layer}-ov{self.overlap}"
        )

    def __repr__(self):
        return f"MegaDNA(merging_strategy={self.merging_strategy}, overlap={self.overlap}, get_layer='{self.get_layer}')"
