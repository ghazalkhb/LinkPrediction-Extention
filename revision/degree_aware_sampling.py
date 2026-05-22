"""
Degree-aware negative sampling for revision experiments.

Matches the original paper's `advanced_negative_sampling` strategy:
source nodes are sampled proportional to degree^alpha, destinations
are sampled uniformly. Forbidden edges (and their reverses) are excluded.

Provides two interfaces:
  - sample_negative_edges: uses a Python set of forbidden (src, dst) tuples
  - sample_negative_edges_fast: uses a numpy array of forbidden keys (src*num_nodes+dst)
"""

import random
from typing import Set, Tuple

import numpy as np
import torch


def _compute_degree_probs(edge_index: torch.Tensor, num_nodes: int, alpha: float = 0.1) -> np.ndarray:
    """Compute degree-biased sampling probabilities for source nodes."""
    if edge_index.numel() == 0:
        # Uniform fallback when no edges exist
        return np.ones(num_nodes, dtype=np.float64) / num_nodes
    flat = edge_index.cpu().numpy().flatten()
    degrees = np.bincount(flat, minlength=num_nodes).astype(np.float64)
    # Avoid zero probabilities: add small epsilon
    degrees = degrees + 1e-8
    prob = np.power(degrees / degrees.sum(), alpha)
    prob = prob / prob.sum()
    return prob


def sample_negative_edges(
    num_nodes: int,
    count: int,
    forbidden: Set[Tuple[int, int]],
    seed: int,
    edge_index: torch.Tensor = None,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Degree-aware negative sampling using set-based forbidden edges.

    Parameters
    ----------
    num_nodes : int
    count : int - number of negatives to sample
    forbidden : set of (src, dst) tuples to exclude (also excludes reverse)
    seed : int
    edge_index : torch.Tensor [2, E] - edges for degree computation.
                 If None, falls back to uniform sampling.
    alpha : float - degree exponent (default 0.1 as in the original paper)
    """
    if count <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    rng = random.Random(seed)

    if edge_index is not None and edge_index.numel() > 0:
        probs = _compute_degree_probs(edge_index, num_nodes, alpha)
        # Build CDF for manual sampling with stdlib Random
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0  # ensure exact
    else:
        probs = None
        cdf = None

    negatives: list[Tuple[int, int]] = []
    seen = set(forbidden)

    max_trials = max(10000, count * 100)
    trials = 0
    while len(negatives) < count and trials < max_trials:
        # Source: degree-aware
        if cdf is not None:
            r = rng.random()
            s = int(np.searchsorted(cdf, r))
            if s >= num_nodes:
                s = num_nodes - 1
        else:
            s = rng.randrange(num_nodes)

        # Destination: uniform
        d = rng.randrange(num_nodes)
        if s == d:
            trials += 1
            continue
        if (s, d) in seen or (d, s) in seen:
            trials += 1
            continue
        seen.add((s, d))
        negatives.append((s, d))
        trials += 1

    # Fallback if we couldn't get enough
    if len(negatives) < count:
        for s in range(num_nodes):
            if len(negatives) >= count:
                break
            for d in range(num_nodes):
                if s == d:
                    continue
                if (s, d) in seen or (d, s) in seen:
                    continue
                seen.add((s, d))
                negatives.append((s, d))
                if len(negatives) >= count:
                    break

    if len(negatives) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(negatives, dtype=torch.long).t().contiguous()


def sample_negative_edges_fast(
    num_nodes: int,
    count: int,
    forbidden_keys: np.ndarray,
    seed: int,
    edge_index: torch.Tensor = None,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Degree-aware negative sampling using numpy key-based forbidden edges.

    Parameters
    ----------
    num_nodes : int
    count : int - number of negatives to sample
    forbidden_keys : np.ndarray - array of keys (src*num_nodes+dst) to exclude
    seed : int
    edge_index : torch.Tensor [2, E] - edges for degree computation.
                 If None, falls back to uniform sampling.
    alpha : float - degree exponent (default 0.1 as in the original paper)
    """
    if count <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    rng = np.random.default_rng(seed)

    if edge_index is not None and edge_index.numel() > 0:
        probs = _compute_degree_probs(edge_index, num_nodes, alpha)
    else:
        probs = None

    collected_src: list[np.ndarray] = []
    collected_dst: list[np.ndarray] = []
    remaining = count

    while remaining > 0:
        batch = int(max(2048, remaining * 2))

        # Source: degree-aware sampling
        if probs is not None:
            src = rng.choice(num_nodes, size=batch, p=probs).astype(np.int64)
        else:
            src = rng.integers(0, num_nodes, size=batch, dtype=np.int64)

        # Destination: uniform
        dst = rng.integers(0, num_nodes, size=batch, dtype=np.int64)

        mask = src != dst
        if forbidden_keys.size > 0:
            keys = src * num_nodes + dst
            mask = mask & (~np.isin(keys, forbidden_keys, assume_unique=False))

        valid_src = src[mask]
        valid_dst = dst[mask]
        if valid_src.size == 0:
            continue

        take = min(remaining, valid_src.size)
        collected_src.append(valid_src[:take])
        collected_dst.append(valid_dst[:take])
        remaining -= take

    neg_np = np.vstack([np.concatenate(collected_src), np.concatenate(collected_dst)])
    return torch.from_numpy(neg_np).to(dtype=torch.long)
