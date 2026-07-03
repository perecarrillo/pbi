import gc
import torch


def clean_gpu() -> None:
    """Free unused GPU memory and run Python's garbage collector."""
    torch.cuda.empty_cache()
    gc.collect()
