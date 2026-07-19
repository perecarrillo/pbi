"""
K-mer frequency feature computation.

Provides fast, vectorized calculation of relative k-mer frequencies from DNA sequences.
K-mer features serve as a complementary, explicit representation to dense neural embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List

from pbi_utils.logging import Logging

logger = Logging()

# Lookup table mapping ASCII byte values (0-255) to base-4 digits: A=0, C=1, G=2, T=3.
# All other characters (e.g. N, U, -, or unknown) map to -1.
_ASCII_TO_BASE4 = np.full(256, -1, dtype=np.int8)
for _char, _val in [('A', 0), ('C', 1), ('G', 2), ('T', 3), ('a', 0), ('c', 1), ('g', 2), ('t', 3)]:
    _ASCII_TO_BASE4[ord(_char)] = _val


def compute_kmer_features(sequence: str, k_values: List[int]) -> np.ndarray:
    """
    Compute a concatenated vector of relative canonical k-mer frequencies for a DNA sequence.

    For each k in ``k_values``, all canonical 4^k k-mers (over A, C, G, T in lexicographical order)
    are counted from sliding windows of size k. K-mers containing non-ACGT characters (such as N)
    are skipped. The counts for each k are normalized to relative frequencies summing to 1.0.
    If no valid k-mers of length k exist in the sequence, a zero vector of size 4^k is returned for that k.

    :param sequence: DNA sequence string.
    :param k_values: List of k values (e.g., [3, 4, 5]).
    :return: 1-D numpy array of shape (sum(4^k for k in k_values),) with dtype float32.
    """
    if not sequence:
        total_dim = sum(4**k for k in k_values)
        return np.zeros(total_dim, dtype=np.float32)

    # Convert sequence string to ASCII byte array and map to base-4 integers (-1 to 3)
    seq_bytes = np.frombuffer(sequence.encode('ascii', 'ignore'), dtype=np.uint8)
    mapped_seq = _ASCII_TO_BASE4[seq_bytes]
    seq_len = len(mapped_seq)

    # Pre-identify valid ACGT bases (>= 0)
    valid_bases = (mapped_seq >= 0)
    # Clean sequence for safe integer arithmetic (replace -1 with 0)
    clean_seq = np.where(valid_bases, mapped_seq, 0).astype(np.int32)

    feature_vectors = []
    for k in k_values:
        dim = 4**k
        if seq_len < k:
            feature_vectors.append(np.zeros(dim, dtype=np.float32))
            continue

        # Check validity of sliding windows of size k using 1D boolean reduction over window view
        windows_valid = np.all(np.lib.stride_tricks.sliding_window_view(valid_bases, k), axis=1)

        if not np.any(windows_valid):
            feature_vectors.append(np.zeros(dim, dtype=np.float32))
            continue

        # Compute base-4 integer index using 1D rolling summation without creating 2D matrices
        num_windows = seq_len - k + 1
        indices = np.zeros(num_windows, dtype=np.int32)
        for j in range(k):
            power = int(4 ** (k - 1 - j))
            indices += clean_seq[j : num_windows + j] * power

        # Only count valid windows
        valid_indices = indices[windows_valid]
        counts = np.bincount(valid_indices, minlength=dim).astype(np.float32)
        total = counts.sum()
        if total > 0:
            counts /= total
        feature_vectors.append(counts)

    return np.concatenate(feature_vectors, axis=0)


def compute_kmer_features_batch(
    sequences: List[str],
    k_values: List[int],
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Compute k-mer feature tensors for a batch of sequences.

    :param sequences: List of DNA sequence strings.
    :param k_values: List of k values (e.g., [3, 4, 5]).
    :param device: Target PyTorch device (e.g., "cpu" or "cuda:0").
    :return: List of 1-D float32 PyTorch tensors on the specified device.
    """
    tensors = []
    for seq in sequences:
        features = compute_kmer_features(seq, k_values)
        tensors.append(torch.tensor(features, dtype=torch.float32, device=device))
    return tensors
