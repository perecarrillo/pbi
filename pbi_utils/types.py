from typing import Literal

DIMENSIONALITY_REDUCTION_TECHNIQUE = Literal["none", "PCA", "UMAP"]

CACHED_EMBEDDINGS_OPTION = bool | Literal["auto"]
