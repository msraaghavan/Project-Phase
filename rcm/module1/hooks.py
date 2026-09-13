"""Per-head attention taps.

Storing every attention matrix is impossible: a single conversation of T tokens
on an L-layer, H-head model materialises L*H*T^2 floats.  The tap therefore
*reduces as it goes*, keeping only the four statistics the metrics of Module 2
actually need:

``alpha_bar``    head- and layer-averaged attention, the T x T matrix used by
                 the reuse, lifetime and centrality metrics;
``topk_count``   how often each key position sits in the top-k of a query's
                 attention distribution, the numerator of the persistence
                 metric;
``mass``         attention mass received per (layer, query head, position), the
                 per-head signal the reservoir analysis is built on;
``n_queries``    the denominator bookkeeping.

This is the "on-GPU accumulation, asynchronous flush" box of Figure 1: the
accumulators live on the compute device and only the reduced statistics ever
reach the trace store.
"""

from __future__ import annotations

import numpy as np
import torch


class AttentionTap:
    """Streaming accumulator for per-head attention statistics."""

    def __init__(self, n_layers: int, n_heads: int, max_len: int, top_k: int = 8,
                 device: torch.device | str = "cpu") -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.top_k = top_k
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        L, H, M = self.n_layers, self.n_heads, self.max_len
        self.alpha_sum = torch.zeros(M, M, device=self.device)
        self.topk_count = torch.zeros(M, device=self.device)
        self.mass = torch.zeros(L, H, M, device=self.device)
        self.n_observed = 0          # number of (layer, head, query) triples seen
        self.seq_len = 0

    @torch.no_grad()
    def observe(self, layer_idx: int, attn: torch.Tensor, q_start: int) -> None:
        """Accumulate one layer's attention block.

        ``attn`` is ``[n_heads, n_q, n_k]`` with rows summing to one, where the
        query at row ``i`` sits at absolute position ``q_start + i`` and the key
        at column ``j`` at absolute position ``j``.
        """
        attn = attn.detach()
        n_heads, n_q, n_k = attn.shape
        q_end = q_start + n_q
        if q_end > self.max_len or n_k > self.max_len:
            raise ValueError(
                f"sequence exceeds tap capacity ({q_end} > {self.max_len}); "
                "raise model.max_seq_len"
            )
        self.seq_len = max(self.seq_len, n_k)

        # head/layer-averaged attention (divided by L*H in `finalize`)
        self.alpha_sum[q_start:q_end, :n_k] += attn.sum(dim=0)
        self.mass[layer_idx, :, :n_k] += attn.sum(dim=1)

        # top-k membership, counted only for strictly earlier keys (q > t)
        k = min(self.top_k, n_k)
        idx = attn.topk(k, dim=-1).indices                       # [H, n_q, k]
        flat = torch.zeros_like(attn)
        flat.scatter_(-1, idx, 1.0)
        q_abs = torch.arange(q_start, q_end, device=attn.device).view(1, n_q, 1)
        k_abs = torch.arange(n_k, device=attn.device).view(1, 1, n_k)
        flat = flat * (k_abs < q_abs)
        self.topk_count[:n_k] += flat.sum(dim=(0, 1))
        self.n_observed += n_heads * n_q

    def finalize(self, seq_len: int | None = None) -> dict[str, np.ndarray]:
        """Return the reduced statistics for a completed conversation."""
        T = int(seq_len or self.seq_len)
        denom = self.n_layers * self.n_heads
        alpha_bar = (self.alpha_sum[:T, :T] / denom).cpu().numpy()
        return {
            "alpha_bar": alpha_bar.astype(np.float32),
            "topk_count": self.topk_count[:T].cpu().numpy().astype(np.float32),
            "mass": self.mass[:, :, :T].cpu().numpy().astype(np.float32),
            "seq_len": T,
            "top_k": self.top_k,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
        }
