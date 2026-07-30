from abc import ABC, abstractmethod
import torch
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy
from pbi_utils.utils import clean_gpu


class AbstractModel(ABC):
    """
    Abstract class for DNA sequence embedding models. All embedding models should inherit from this class and implement the _compute_single_embedding and _encode methods.
    """

    @abstractmethod
    def __init__(
        self,
        max_seq_len: int,
        merging_strategy: AbstractMergerStrategy = TruncateStrategy(),
        device="cpu",
        overlap: int = 0,
        load_model: bool = True,
        batch_size: int = 1,
    ) -> None:
        """Abstract class for DNA sequence embedding models.
        Args:
            max_seq_len (int): Maximum sequence length for the model.
            merging_strategy (AbstractMergerStrategy, optional): Strategy to merge embeddings from subsequences. Defaults to TruncateStrategy().
            overlap (int, optional): Overlap size between subsequences. Defaults to 0.
            load_model (bool, optional): Whether to load the model for embedding computation. Defaults to True. If False, the embed method will raise an error, and should only be used to get the class name.
            batch_size (int, optional): Batch size for embedding computation. Defaults to 1.
        """
        self.merging_strategy = merging_strategy
        self.overlap = int(overlap)
        self.max_seq_len = int(float(max_seq_len))
        self.load_model = load_model
        self.batch_size = batch_size
        self.device = device
        super().__init__()

    def embed_raw(self, dna_sequence: str) -> torch.Tensor:
        """Compute the raw, unmerged subsequence embeddings for a DNA sequence.
        Returns a 2D tensor of shape [N, hidden_dim].
        """
        if not self.load_model:
            raise RuntimeError(
                "Model not loaded. If you want to compute embeddings, please set load_model to True when initializing the class."
            )

        sequences = self._split_sequence(dna_sequence)

        chunk_batch_size = max(16, self.batch_size)
        outputs = []
        for i in range(0, len(sequences), chunk_batch_size):
            seq_batch = sequences[i : i + chunk_batch_size]
            tokens = self._encode(seq_batch)
            batch_embeds = self._compute_batch_embeddings(tokens)
            if batch_embeds.dim() == 3 and batch_embeds.size(1) == 1:
                batch_embeds = batch_embeds.squeeze(1)
            outputs.append(batch_embeds.cpu())
            del tokens, batch_embeds
            clean_gpu()

        embeddings = torch.cat(outputs, dim=0)
        clean_gpu()
        return embeddings

    def embed(self, dna_sequence: str) -> torch.Tensor:
        """Compute the merged embedding for a DNA sequence."""
        sequences = self._split_sequence(dna_sequence)

        # Only keep the required chunks if using truncation strategies
        if self.merging_strategy.name() == "TruncateStrategy":
            sequences = [sequences[0]]
        elif self.merging_strategy.name() == "BottomTruncateStrategy":
            sequences = [sequences[-1]]
        elif self.merging_strategy.name() == "TopBottomTruncateStrategy":
            sequences = [sequences[0], sequences[-1]]

        sub_dna = "".join(sequences)
        raw_embeddings = self.embed_raw(sub_dna)
        merged_embedding = self.merging_strategy.merge(sequences, raw_embeddings)

        clean_gpu()

        return merged_embedding

    def _compute_batch_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        # Batch size is not useful, as it takes almost the same time as doing it one by one and it uses much more memory. Might be useful if model is parallelized

        outputs = []

        with torch.no_grad():
            for i in range(0, tokens.shape[0], self.batch_size):
                batch = tokens[i : i + self.batch_size]
                embeds = self._compute_single_embedding(batch.to(self.device))
                outputs.append(embeds.cpu())
                if i % 10 == 0:
                    clean_gpu()

        clean_gpu()
        return torch.cat(outputs, dim=0)

    # Divide sequence into overlapping subsequences
    def _split_sequence(self, sequence: str) -> list[str]:
        step = self.max_seq_len - self.overlap
        subsequences = [
            sequence[i : i + self.max_seq_len] for i in range(0, len(sequence), step)
        ]
        return subsequences

    @abstractmethod
    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the embedding for a single tokenized sequence.
        Args:
            tokens (torch.Tensor): Tokenized sequence of size [1, self.max_seq_len].
        Returns:
            torch.Tensor: Embedding of the sequence of size [1, embedding_size].
        """
        pass

    @abstractmethod
    def _encode(self, dna_sequences: list[str]) -> torch.Tensor:
        """Encode a list of DNA sequences into token tensors.
        Args:
            dna_sequences (list[str]): List of DNA sequences of size < self.max_seq_len.
        Returns:
            torch.Tensor: Tokenized sequences tensor of size [batch_size, self.max_seq_len].
        """
        pass

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready to compute embeddings."""
        return self.load_model

    def raw_name(self) -> str:
        """Base name identifying the unmerged embedding storage key."""
        return f"{type(self).__name__}-RAW-ov{self.overlap}-maxlen{self.max_seq_len}"

    def name(self) -> str:
        return f"{type(self).__name__}-{self.merging_strategy.name()}"

    def __repr__(self):
        return f"{type(self).__name__}(merging_strategy={self.merging_strategy})"
