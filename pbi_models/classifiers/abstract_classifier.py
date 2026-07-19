import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from abc import ABC, abstractmethod

logger = Logging()


class AbstractNNClassifier(nn.Module, ABC):
    """
    Abstract base class for classifiers based on pytorch. All classifiers should inherit from this class and implement the forward method.

    **Note:** By default, the CrossEntropyLoss already applies Softmax internally, so the model must output raw logits (directly from the last layer). If not, it can cause instabilities and can cause numerical issues.
    """

    @abstractmethod
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, kmer_dim: int = 0):
        super().__init__()
        self.kmer_dim = kmer_dim

    @abstractmethod
    def forward(
        self,
        bacterium_emb: torch.Tensor,
        phage_emb: torch.Tensor,
        kmer_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, bact_emb_dim]
            phage_emb:     [batch, phag_emb_dim]
            kmer_emb:      optional [batch, kmer_dim]
        Returns:
            logits: [batch, num_classes]
        """
        pass

    def reset_model(self, device: str):
        def reset_weights(model: nn.Module):
            """
            Recursively reset the weights of a PyTorch model in-place.
            Works for nested submodules, Sequential, and custom modules.
            Coauthored by ChatGPT
            """
            for child in model.children():
                # If the layer has reset_parameters, call it
                if hasattr(child, "reset_parameters"):
                    child.reset_parameters()
                else:
                    # Recursively reset nested children
                    reset_weights(child)

        # Reset model for each fold
        reset_weights(self)
        self.to(device)

    def name(self) -> str:
        return f"{type(self).__name__}"

    def __repr__(self):
        return f"{type(self).__name__}()"
