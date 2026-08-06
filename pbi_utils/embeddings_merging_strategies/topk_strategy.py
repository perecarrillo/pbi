from typing import Literal
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
import torch
import torch.nn.functional as F


class TopKStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by keeping only the K chunks most similar to the mean
    chunk embedding (cosine similarity), then averaging or concatenating those K chunks.

    Intuition: subsequences that deviate strongly from the centroid of the
    genome may be noisy or low-quality; keeping only the most *representative*
    ones acts as an unsupervised quality filter.

    When a sequence has fewer than K chunks, all chunks are used and padded/repeated
    if merging_strategy is "concat" to guarantee a consistent output dimension of
    ``[1, K * dim]``.

    :param K: Number of top chunks to keep.
    :type K: int
    :param merging_strategy: Mode for combining the K selected chunks:
        "concat" to concatenate into shape ``[1, K * dim]``,
        or "avg" to average into shape ``[1, dim]`` (default).
    :type merging_strategy: Literal["concat", "avg"]
    """

    def __init__(
        self,
        K: int = 5,
        merging_strategy: Literal["concat", "avg"] = "avg",
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.merging_strategy = merging_strategy

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        """
        :param sentences: List of DNA sub-sequences (not used for scoring).
        :param embeddings: Tensor of shape ``[N, dim]``.
        :return: Merged embedding of shape ``[1, K * dim]`` (concat) or ``[1, dim]`` (avg).
        """
        n = embeddings.shape[0]
        k = min(self.K, n)

        # Compute mean chunk embedding and cosine similarities
        mean_embed = embeddings.mean(dim=0, keepdim=True)  # [1, dim]
        # Normalize both
        norm_embeds = F.normalize(embeddings, dim=1)          # [N, dim]
        norm_mean = F.normalize(mean_embed, dim=1)            # [1, dim]
        similarities = (norm_embeds * norm_mean).sum(dim=1)   # [N]

        # Select top-K indices (most similar to centroid)
        topk_idx = torch.topk(similarities, k=k, largest=True).indices  # [k]
        selected = embeddings[topk_idx]                                 # [k, dim]

        if self.merging_strategy == "concat":
            # If fewer than K chunks, pad by repeating selected chunks up to K
            if k < self.K:
                repeat_factor = (self.K + k - 1) // k
                selected = selected.repeat(repeat_factor, 1)[: self.K]          # [K, dim]
            return selected.reshape(1, -1)                                      # [1, K * dim]
        else:
            return selected.mean(dim=0, keepdim=True)                           # [1, dim]

    def name(self) -> str:
        if self.merging_strategy == "concat":
            return f"TopK-{self.merging_strategy}-K{self.K}"
        return f"TopK-K{self.K}"

    def __repr__(self) -> str:
        return f"TopKStrategy(K={self.K}, merging_strategy='{self.merging_strategy}')"
