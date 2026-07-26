import ast
import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractNNClassifier

logger = Logging()


class MLPBlock(nn.Module):
    """Simple MLP + ReLU + Dropout block."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()

        self.fc = nn.Linear(in_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class BranchMLP(nn.Module):
    """
    Branch MLP for either bacteria or phage input.
    """

    def __init__(
        self, mlp_params: list[int], initial_input_size: int, dropout: float = 0.2
    ):
        """
        mlp_params: list of hidden layer sizes
        Example: [128, 64, 32]
        """
        super().__init__()
        layers = []
        in_size = initial_input_size
        for hid in mlp_params:
            layers.append(MLPBlock(in_size, int(hid), dropout))
            in_size = int(hid)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPClassifier(AbstractNNClassifier):
    """
    General MLP classifier with two branches (one for bacteria, one for phages) plus
    an optional dedicated branch for k-mer features.
    """

    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        bacterium_mlp_sizes: list[int] | str,
        phage_mlp_sizes: list[int] | str,
        dropout: float | str = 0.2,
        dense_dim: int | str = 128,
        kmer_dim: int = 0,
        kmer_mlp_sizes: list[int] | str = [],
    ):

        super().__init__(bacterium_embed_dim, phage_embed_dim, kmer_dim)

        # Sanity checks
        self._sanity_checks(
            bacterium_embed_dim,
            phage_embed_dim,
            bacterium_mlp_sizes,
            phage_mlp_sizes,
            dense_dim,
            kmer_dim,
            kmer_mlp_sizes,
        )

        # Convert params to dict
        if isinstance(bacterium_mlp_sizes, str):
            bacterium_mlp_sizes = self._parse_branch_params(bacterium_mlp_sizes)
        if isinstance(phage_mlp_sizes, str):
            phage_mlp_sizes = self._parse_branch_params(phage_mlp_sizes)
        if isinstance(kmer_mlp_sizes, str):
            kmer_mlp_sizes = self._parse_branch_params(kmer_mlp_sizes)

        self.bacterium_mlp_sizes = bacterium_mlp_sizes
        self.phage_mlp_sizes = phage_mlp_sizes
        self.kmer_mlp_sizes = kmer_mlp_sizes

        # Branches
        self.bacteria_branch = BranchMLP(
            bacterium_mlp_sizes, bacterium_embed_dim, float(dropout)
        )
        self.phage_branch = BranchMLP(phage_mlp_sizes, phage_embed_dim, float(dropout))
        if kmer_dim > 0:
            self.kmer_branch = BranchMLP(kmer_mlp_sizes, kmer_dim, float(dropout))
        else:
            self.kmer_branch = None

        # Compute flattened size
        bacterium_flat_size = (
            bacterium_mlp_sizes[-1]
            if len(bacterium_mlp_sizes) > 0
            else bacterium_embed_dim
        )
        phage_flat_size = (
            phage_mlp_sizes[-1] if len(phage_mlp_sizes) > 0 else phage_embed_dim
        )
        kmer_flat_size = (
            (kmer_mlp_sizes[-1] if len(kmer_mlp_sizes) > 0 else kmer_dim)
            if kmer_dim > 0
            else 0
        )
        concat_dim = bacterium_flat_size + phage_flat_size + kmer_flat_size

        # Dense layers
        self.fc1 = nn.Linear(concat_dim, int(dense_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(dense_dim), 2)

    def _sanity_checks(
        self,
        bacterium_embed_dim,
        phage_embed_dim,
        bacterium_mlp_sizes,
        phage_mlp_sizes,
        dense_dim,
        kmer_dim=0,
        kmer_mlp_sizes=[],
    ):
        if isinstance(bacterium_mlp_sizes, list):
            for tpl in bacterium_mlp_sizes:
                assert isinstance(
                    tpl, int
                ), f"bacterium_conv_params must be a list of integers or a string. Got: {bacterium_mlp_sizes}"
        else:
            assert isinstance(
                bacterium_mlp_sizes, str
            ), f"bacterium_conv_params must be either a list of integers or a string. Got: {bacterium_mlp_sizes}"
        if isinstance(phage_mlp_sizes, list):
            for tpl in phage_mlp_sizes:
                assert isinstance(
                    tpl, int
                ), f"phage_conv_params must be a list of integers or a string. Got: {phage_mlp_sizes}"
        else:
            assert isinstance(
                phage_mlp_sizes, str
            ), f"phage_conv_params must be either a list of integers or a string. Got: {phage_mlp_sizes}"
        if isinstance(kmer_mlp_sizes, list):
            for tpl in kmer_mlp_sizes:
                assert isinstance(
                    tpl, int
                ), f"kmer_mlp_sizes must be a list of integers or a string. Got: {kmer_mlp_sizes}"
        else:
            assert isinstance(
                kmer_mlp_sizes, str
            ), f"kmer_mlp_sizes must be either a list of integers or a string. Got: {kmer_mlp_sizes}"
        assert isinstance(
            dense_dim, (int, str)
        ), f"dense_dim must be either an integer or a string. Got: {type(dense_dim)}: {dense_dim}"
        assert isinstance(
            bacterium_embed_dim, int
        ), f"bacterium_embed_dim must be an integer. Got {type(bacterium_embed_dim)}: {bacterium_embed_dim}"
        assert isinstance(
            phage_embed_dim, int
        ), f"phage_embed_dim must be an integer. Got {type(phage_embed_dim)}: {phage_embed_dim}"
        assert isinstance(
            kmer_dim, int
        ), f"kmer_dim must be an integer. Got {type(kmer_dim)}: {kmer_dim}"

    def _parse_branch_params(self, params_str: str) -> list[int]:
        """Parse branch parameters from string representation."""
        # Using ast.literal_eval for safe evaluation
        return ast.literal_eval(params_str)

    def forward(
        self,
        bacterium_emb: torch.Tensor,
        phage_emb: torch.Tensor,
        kmer_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
            kmer_emb:      optional [batch, kmer_dim]
        Returns:
            logits: [batch, num_classes]
        """

        # Branch processing
        x_b = self.bacteria_branch(bacterium_emb)
        x_p = self.phage_branch(phage_emb)

        # Flatten & concatenate
        x_b = torch.flatten(x_b, start_dim=1)
        x_p = torch.flatten(x_p, start_dim=1)
        if self.kmer_branch is not None and kmer_emb is not None:
            x_k = self.kmer_branch(kmer_emb)
            x_k = torch.flatten(x_k, start_dim=1)
            x = torch.cat((x_b, x_p, x_k), dim=1)
        else:
            x = torch.cat((x_b, x_p), dim=1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # No softmax, return raw logits
        return x

    def name(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        kmer_part = f", kmer_mlp_sizes={self.kmer_mlp_sizes}" if self.kmer_dim > 0 else ""
        return f"""MLPClassifier(bacterium_mlp_sizes={self.bacterium_mlp_sizes}, phage_mlp_sizes={self.phage_mlp_sizes}{kmer_part}), dense_dim={self.fc1.out_features}), dropout={self.dropout.p})"""


class BasicMLPClassifier(MLPClassifier):
    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        mlp_params: list[int] | str,
        dropout: float | str = 0.5,
        kmer_dim: int = 0,
        kmer_mlp_sizes: list[int] | str = [],
    ):
        super().__init__(
            bacterium_embed_dim,
            phage_embed_dim,
            [],
            [],
            dropout,
            0,
            kmer_dim,
            kmer_mlp_sizes,
        )

        if isinstance(mlp_params, str):
            self.params = self._parse_branch_params(mlp_params)
        else:
            self.params = mlp_params

        self.mlp = BranchMLP(
            self.params, bacterium_embed_dim + phage_embed_dim, float(dropout)
        )

        if isinstance(kmer_mlp_sizes, str):
            kmer_mlp_sizes = self._parse_branch_params(kmer_mlp_sizes)
        self.kmer_mlp_sizes = kmer_mlp_sizes

        if kmer_dim > 0:
            self.kmer_branch = BranchMLP(kmer_mlp_sizes, kmer_dim, float(dropout))
            kmer_flat = kmer_mlp_sizes[-1] if len(kmer_mlp_sizes) > 0 else kmer_dim
        else:
            self.kmer_branch = None
            kmer_flat = 0

        mlp_out = self.params[-1] if len(self.params) > 0 else bacterium_embed_dim + phage_embed_dim
        self.fc2 = nn.Linear(mlp_out + kmer_flat, 2)

    def forward(
        self,
        bacterium_emb: torch.Tensor,
        phage_emb: torch.Tensor,
        kmer_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
            kmer_emb:      optional [batch, kmer_dim]
        Returns:
            logits: [batch, num_classes]
        """

        x = torch.cat((bacterium_emb, phage_emb), dim=1)

        x = self.mlp(x)

        x = torch.flatten(x, start_dim=1)

        if self.kmer_branch is not None and kmer_emb is not None:
            x_k = self.kmer_branch(kmer_emb)
            x_k = torch.flatten(x_k, start_dim=1)
            x = torch.cat((x, x_k), dim=1)

        x = self.fc2(x)
        # No softmax, return raw logits
        return x

    def __repr__(self) -> str:
        kmer_part = f", kmer_mlp_sizes={self.kmer_mlp_sizes}" if self.kmer_dim > 0 else ""
        return f"""BasicMLPClassifier(mlp_params: {self.params}{kmer_part}, dropout={self.dropout.p})"""
