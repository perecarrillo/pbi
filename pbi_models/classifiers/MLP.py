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
    General MLP classifier with two branches (one for bacteria, one for phages).
    """

    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        bacterium_mlp_sizes: list[int] | str,
        phage_mlp_sizes: list[int] | str,
        dropout: float | str = 0.2,
        dense_dim: int | str = 128,
    ):

        super().__init__(bacterium_embed_dim, phage_embed_dim)

        # Sanity checks
        self._sanity_checks(
            bacterium_embed_dim,
            phage_embed_dim,
            bacterium_mlp_sizes,
            phage_mlp_sizes,
            dense_dim,
        )

        # Convert params to dict
        if isinstance(bacterium_mlp_sizes, str):
            bacterium_mlp_sizes = self._parse_branch_params(bacterium_mlp_sizes)
        if isinstance(phage_mlp_sizes, str):
            phage_mlp_sizes = self._parse_branch_params(phage_mlp_sizes)

        self.bacterium_mlp_sizes = bacterium_mlp_sizes
        self.phage_mlp_sizes = phage_mlp_sizes

        # Branches
        self.bacteria_branch = BranchMLP(
            bacterium_mlp_sizes, bacterium_embed_dim, float(dropout)
        )
        self.phage_branch = BranchMLP(phage_mlp_sizes, phage_embed_dim, float(dropout))

        # Compute flattened size
        bacterium_flat_size = (
            bacterium_mlp_sizes[-1]
            if len(bacterium_mlp_sizes) > 0
            else bacterium_embed_dim
        )
        phage_flat_size = (
            phage_mlp_sizes[-1] if len(phage_mlp_sizes) > 0 else phage_embed_dim
        )
        concat_dim = bacterium_flat_size + phage_flat_size

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
        assert isinstance(
            dense_dim, (int, str)
        ), f"dense_dim must be either an integer or a string. Got: {type(dense_dim)}: {dense_dim}"
        assert isinstance(
            bacterium_embed_dim, int
        ), f"bacterium_embed_dim must be an integer. Got {type(bacterium_embed_dim)}: {bacterium_embed_dim}"
        assert isinstance(
            phage_embed_dim, int
        ), f"phage_embed_dim must be an integer. Got {type(phage_embed_dim)}: {phage_embed_dim}"

    def _parse_branch_params(self, params_str: str) -> list[int]:
        """Parse branch parameters from string representation."""
        # Using ast.literal_eval for safe evaluation
        return ast.literal_eval(params_str)

    def forward(
        self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """

        # Branch processing
        x_b = self.bacteria_branch(bacterium_emb)
        x_p = self.phage_branch(phage_emb)

        # Flatten & concatenate
        x_b = torch.flatten(x_b, start_dim=1)
        x_p = torch.flatten(x_p, start_dim=1)
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
        return f"""MLPClassifier(bacterium_mlp_sizes={self.bacterium_mlp_sizes}, phage_mlp_sizes={self.phage_mlp_sizes}), dense_dim={self.fc1.out_features}), dropout={self.dropout.p})"""


class BasicMLPClassifier(MLPClassifier):
    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        mlp_params: list[int] | str,
        dropout: float | str = 0.5,
    ):
        super().__init__(bacterium_embed_dim, phage_embed_dim, [], [], dropout, 0)

        if isinstance(mlp_params, str):
            self.params = self._parse_branch_params(mlp_params)
        else:
            self.params = mlp_params

        self.mlp = BranchMLP(
            self.params, bacterium_embed_dim + phage_embed_dim, float(dropout)
        )

        self.fc2 = nn.Linear(self.params[-1], 2)

    def forward(
        self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """

        x = torch.cat((bacterium_emb, phage_emb), dim=1)

        x = self.mlp(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc2(x)
        # No softmax, return raw logits
        return x

    def __repr__(self) -> str:
        return f"""MLPClassifier(mlp_params: {self.params}, dropout={self.dropout.p})"""


class ResidualBranchMLP(nn.Module):
    """
    Branch MLP with residual skip connections between layers.

    Each layer applies ``Linear -> LayerNorm/BatchNorm -> ReLU -> Dropout`` and adds a residual shortcut: ``output = layer(x) + project(x)``.  The shortcut is a ``Linear`` projection when input and output dimensions differ, or ``Identity`` when they are the same.
    """

    def __init__(
        self,
        mlp_sizes: list[int],
        input_dim: int,
        dropout: float,
        use_layernorm: bool = True,
    ):
        """
        :param mlp_sizes: Hidden layer sizes.
        :param input_dim: Dimensionality of the input features.
        :param dropout: Dropout probability applied after each layer.
        :param use_layernorm: Use LayerNorm (True) or BatchNorm1d (False) after the linear projection.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()

        in_dim = input_dim
        for out_dim in mlp_sizes:
            out_dim = int(out_dim)
            norm = nn.LayerNorm(out_dim) if use_layernorm else nn.BatchNorm1d(out_dim)
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    norm,
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            # project only when dimensions change
            if in_dim != out_dim:
                self.projections.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.projections.append(nn.Identity())
            in_dim = out_dim

        self.out_dim = in_dim  # dimension of the final output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, proj in zip(self.layers, self.projections):
            x = layer(x) + proj(x)
        return x


class AttentionMLPClassifier(AbstractNNClassifier):
    """
    Two-branch MLP classifier with a scalar cross-attention gate.

    Architecture:
    1. Two :class:`ResidualBranchMLP` branches (one per organism) each project to
       ``shared_dim = max(bact_out, phag_out)``.
    2. A **scalar attention gate** lets the bacterium embedding query the phage
       embedding: ``weight = sigmoid(dot(x_b, x_p) / sqrt(shared_dim))`` then
       ``attn_out = weight * x_p``.
    3. Concatenate ``[x_b, x_p, attn_out, x_b - attn_out]`` -> 4 * shared_dim.
    4. ``Linear(4*shared_dim, dense_dim) -> ReLU -> LayerNorm -> Dropout -> Linear(dense_dim, 2)``
    """

    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        bacterium_mlp_sizes: list[int] | str,
        phage_mlp_sizes: list[int] | str,
        dropout: float | str = 0.2,
        dense_dim: int | str = 128,
        use_layernorm: bool = True,
    ):
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        # Parse string params
        if isinstance(bacterium_mlp_sizes, str):
            bacterium_mlp_sizes = ast.literal_eval(bacterium_mlp_sizes)
        if isinstance(phage_mlp_sizes, str):
            phage_mlp_sizes = ast.literal_eval(phage_mlp_sizes)
        dropout = float(dropout)
        dense_dim = int(dense_dim)

        self._bacterium_mlp_sizes = bacterium_mlp_sizes
        self._phage_mlp_sizes = phage_mlp_sizes
        self._dropout_p = dropout
        self._dense_dim = dense_dim

        # Residual branches
        self.bacteria_branch = ResidualBranchMLP(
            bacterium_mlp_sizes, bacterium_embed_dim, dropout, use_layernorm
        )
        self.phage_branch = ResidualBranchMLP(
            phage_mlp_sizes, phage_embed_dim, dropout, use_layernorm
        )

        bact_out = self.bacteria_branch.out_dim
        phag_out = self.phage_branch.out_dim
        shared_dim = max(bact_out, phag_out)
        self._shared_dim = shared_dim

        # projections to shared_dim
        self.bact_proj = (
            nn.Linear(bact_out, shared_dim, bias=False)
            if bact_out != shared_dim
            else nn.Identity()
        )
        self.phag_proj = (
            nn.Linear(phag_out, shared_dim, bias=False)
            if phag_out != shared_dim
            else nn.Identity()
        )

        # Head: 4 * shared_dim -> dense_dim -> 2
        self.head = nn.Sequential(
            nn.Linear(4 * shared_dim, dense_dim),
            nn.ReLU(),
            nn.LayerNorm(dense_dim),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 2),
        )

    def forward(
        self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        :param bacterium_emb: ``[batch, bacterium_embed_dim]``
        :param phage_emb: ``[batch, phage_embed_dim]``
        :return: Logits ``[batch, 2]``
        """
        x_b = self.bact_proj(self.bacteria_branch(bacterium_emb))
        x_p = self.phag_proj(self.phage_branch(phage_emb))

        # Scalar cross-attention gate, to allow bacteria to query phages
        scale = self._shared_dim ** 0.5
        score = (x_b * x_p).sum(dim=-1, keepdim=True) / scale  # [batch, 1]
        weight = torch.sigmoid(score)
        attn_out = weight * x_p

        combined = torch.cat([x_b, x_p, attn_out, x_b - attn_out], dim=-1)
        return self.head(combined)

    def name(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            f"AttentionMLPClassifier("
            f"bacterium_mlp_sizes={self._bacterium_mlp_sizes}, "
            f"phage_mlp_sizes={self._phage_mlp_sizes}, "
            f"dense_dim={self._dense_dim}, "
            f"dropout={self._dropout_p})"
        )


class BilinearMLPClassifier(AbstractNNClassifier):
    """
    Two-branch MLP classifier using ``nn.Bilinear`` as a pairwise interaction layer.

    Architecture:
    1. Two :class:`ResidualBranchMLP` branches -> project to ``shared_dim``.
    2. ``interaction = ReLU(Bilinear(x_b, x_p))`` -> ``n_factors`` features.
    3. Concatenate ``[x_b, x_p, interaction]`` -> 2 * shared_dim + n_factors.
    4. ``Linear -> ReLU -> LayerNorm -> Dropout -> Linear -> logits``
    """

    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        bacterium_mlp_sizes: list[int] | str,
        phage_mlp_sizes: list[int] | str,
        dropout: float | str = 0.2,
        dense_dim: int | str = 128,
        n_factors: int | str = 64,
        use_layernorm: bool = True,
    ):
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        if isinstance(bacterium_mlp_sizes, str):
            bacterium_mlp_sizes = ast.literal_eval(bacterium_mlp_sizes)
        if isinstance(phage_mlp_sizes, str):
            phage_mlp_sizes = ast.literal_eval(phage_mlp_sizes)
        dropout = float(dropout)
        dense_dim = int(dense_dim)
        n_factors = int(n_factors)

        self._bacterium_mlp_sizes = bacterium_mlp_sizes
        self._phage_mlp_sizes = phage_mlp_sizes
        self._dropout_p = dropout
        self._dense_dim = dense_dim
        self._n_factors = n_factors

        self.bacteria_branch = ResidualBranchMLP(
            bacterium_mlp_sizes, bacterium_embed_dim, dropout, use_layernorm
        )
        self.phage_branch = ResidualBranchMLP(
            phage_mlp_sizes, phage_embed_dim, dropout, use_layernorm
        )

        bact_out = self.bacteria_branch.out_dim
        phag_out = self.phage_branch.out_dim
        shared_dim = max(bact_out, phag_out)
        self._shared_dim = shared_dim

        self.bact_proj = (
            nn.Linear(bact_out, shared_dim, bias=False)
            if bact_out != shared_dim
            else nn.Identity()
        )
        self.phag_proj = (
            nn.Linear(phag_out, shared_dim, bias=False)
            if phag_out != shared_dim
            else nn.Identity()
        )

        self.bilinear = nn.Bilinear(shared_dim, shared_dim, n_factors)

        concat_dim = 2 * shared_dim + n_factors
        self.head = nn.Sequential(
            nn.Linear(concat_dim, dense_dim),
            nn.ReLU(),
            nn.LayerNorm(dense_dim),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 2),
        )

    def forward(
        self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        :param bacterium_emb: ``[batch, bacterium_embed_dim]``
        :param phage_emb: ``[batch, phage_embed_dim]``
        :return: Logits ``[batch, 2]``
        """
        x_b = self.bact_proj(self.bacteria_branch(bacterium_emb))
        x_p = self.phag_proj(self.phage_branch(phage_emb))
        interaction = F.relu(self.bilinear(x_b, x_p))
        combined = torch.cat([x_b, x_p, interaction], dim=-1)
        return self.head(combined)

    def name(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            f"BilinearMLPClassifier("
            f"bacterium_mlp_sizes={self._bacterium_mlp_sizes}, "
            f"phage_mlp_sizes={self._phage_mlp_sizes}, "
            f"dense_dim={self._dense_dim}, "
            f"n_factors={self._n_factors}, "
            f"dropout={self._dropout_p})"
        )

