import torch
from torch import nn
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractNNClassifier

logger = Logging()


class LinearClassifier(AbstractNNClassifier):
    """
    A simple linear classifier that concatenates bacterium and phage embeddings
    (and optionally k-mer features) and passes them through a single linear layer
    to output logits for binary classification.

    Does not perform well, but serves as a baseline model.
    """

    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, kmer_dim: int = 0):
        super().__init__(bacterium_embed_dim, phage_embed_dim, kmer_dim)

        self.linear = nn.Linear(bacterium_embed_dim + phage_embed_dim + kmer_dim, 2)

    def forward(
        self,
        bacterium_emb: torch.Tensor,
        phage_emb: torch.Tensor,
        kmer_emb: torch.Tensor | None = None,
    ):
        """
        Inputs:
            bacterium_emb: [batch, bact_emb_dim]
            phage_emb:     [batch, phag_emb_dim]
            kmer_emb:      optional [batch, kmer_dim]
        Returns:
            logits: [batch, num_classes]
        """
        if kmer_emb is not None and self.kmer_dim > 0:
            x = torch.cat([bacterium_emb, phage_emb, kmer_emb], dim=1)
        else:
            x = torch.cat([bacterium_emb, phage_emb], dim=1)

        x = self.linear(x)

        return x
