from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
from pbi_utils.logging import Logging, logging
from pbi_utils.utils import clean_gpu
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

logger = Logging()


class DNABERT2(AbstractModel):
    def __init__(
        self,
        merging_strategy: AbstractMergerStrategy,
        source_code_path: str,
        device: str = "cpu",
        overlap: int = 0,
        max_seq_len: int = 2**15,
        load_model: bool = True,
    ) -> None:

        self.load_model = load_model
        self.batch_size = 1

        self.device = device
        self.merging_strategy = merging_strategy
        self.overlap = int(overlap)

        if self.load_model:
            # Not show internal transformers logging messages
            current_log_level = logging.root.level
            Logging.set_logging_level()

            self.tokenizer = AutoTokenizer.from_pretrained(
                source_code_path, trust_remote_code=True
            )
            config = BertConfig.from_pretrained(source_code_path)
            self.model = AutoModel.from_pretrained(
                source_code_path, trust_remote_code=True, config=config
            )

            Logging.set_logging_level(current_log_level)

            self.model.to(self.device)
            self.model.eval()

        self.max_seq_len = int(float(max_seq_len))

        logger.debug(f"[DNABERT2] Max sequence length for DNABERT2: {self.max_seq_len}")

    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Compute the embeddings
            output = self.model(tokens)[0]  # [1, sequence_length, 768]

        # embedding with mean pooling
        mean_embed = torch.mean(output[0], dim=0)
        # mean_embed = torch.max(output[0], dim=0)[0]

        mean_embed = mean_embed.unsqueeze(0)  # add batch dimension back

        return mean_embed

    def _encode(self, dna_sequence: list[str]) -> torch.Tensor:
        return self.tokenizer.batch_encode_plus(
            dna_sequence, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )["input_ids"]

    def raw_name(self) -> str:
        return f"DNABERT2-RAW-ov{self.overlap}-maxlen{self.max_seq_len}"

    def name(self) -> str:
        return f"DNABERT2-{self.merging_strategy.name()}-ov{self.overlap}-maxlen{self.max_seq_len}"

    def __repr__(self):
        return f"DNABERT2(merging_strategy={self.merging_strategy}, overlap={self.overlap}, max_seq_len={self.max_seq_len})"
