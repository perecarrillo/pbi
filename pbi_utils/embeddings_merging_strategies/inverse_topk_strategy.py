from typing import Literal
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
import torch
import torch.nn.functional as F


class InverseTopKStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by keeping only the K chunks MOST DISSIMILAR (lowest cosine
    similarity) to the mean chunk embedding, then concatenating (or averaging) those K chunks.

    Intuition: In biological sequences like phage and bacteria genomes, unique,
    divergent, or hyper-variable regions (such as host-binding tail fiber proteins)
    differ significantly from the genome-wide centroid. Concatenating the most divergent
    chunks preserves the exact, distinct feature profiles of all K chunks without blending
    them into a single mean vector.

    When a sequence has fewer than K chunks, chunks are padded/repeated to guarantee a
    consistent output dimension of ``[1, K * dim]`` (or ``[1, dim]`` for avg).

    :param K: Number of most divergent (lowest similarity) chunks to keep.
    :type K: int
    :param merging_strategy: Mode for combining the K selected chunks:
        "concat" to concatenate into shape ``[1, K * dim]`` (default),
        or "avg" to average into shape ``[1, dim]``.
    :type merging_strategy: Literal["concat", "avg"]
    """

    def __init__(
        self,
        K: int = 5,
        merging_strategy: Literal["concat", "avg"] = "concat",
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

        # Select bottom-K (lowest similarity / most divergent) indices
        inverse_topk_idx = torch.topk(similarities, k=k, largest=False).indices  # [k]
        selected = embeddings[inverse_topk_idx]                                  # [k, dim]

        if self.merging_strategy == "concat":
            # If fewer than K chunks, pad by repeating selected chunks up to K
            if k < self.K:
                repeat_factor = (self.K + k - 1) // k
                selected = selected.repeat(repeat_factor, 1)[: self.K]           # [K, dim]
            return selected.reshape(1, -1)                                       # [1, K * dim]
        else:
            return selected.mean(dim=0, keepdim=True)                            # [1, dim]

    def name(self) -> str:
        return f"InverseTopK-{self.merging_strategy}-K{self.K}"

    def __repr__(self) -> str:
        return f"InverseTopKStrategy(K={self.K}, merging_strategy='{self.merging_strategy}')"
