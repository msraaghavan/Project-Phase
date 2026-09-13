"""The provenance-stamped Key-Value cache.

This is the central data structure of the project.  A stock KV cache is an
anonymous tensor: you cannot ask it *which turn*, *which agent* or *which
source channel* an entry came from, so nothing downstream can be attributed.
:class:`TracingCache` stamps every write with

    p_{l,h,t} = (layer, head, absolute position, turn, agent, channel)

which is exactly the provenance record Stage 1 of the methodology requires.

Two implementations share the same stamping logic through
:class:`ProvenanceMixin`:

* :class:`TracingCache` -- self-contained storage, used by the reference
  transformer;
* :class:`~rcm.adapters.hf.HFTracingCache` -- subclasses Hugging Face
  ``DynamicCache`` and delegates tensor storage to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rcm.config import CH_AGENT, CH_PEER


@dataclass(frozen=True)
class KVProvenance:
    """Provenance of one absolute cache position.

    ``layer`` and ``head`` are not stored per position because they are implied
    by the slot the entry occupies; they are materialised when the record is
    written out (see :class:`~rcm.module1.logger.ProvenanceLogger`).
    """

    position: int
    turn_idx: int
    agent_id: str
    channel: str
    token_id: int
    source_position: int | None = None   # original position if transplanted

    def as_row(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "turnIdx": self.turn_idx,
            "agentId": self.agent_id,
            "channel": self.channel,
            "tokenId": self.token_id,
            "sourcePosition": self.source_position,
        }


class ProvenanceMixin:
    """Write-side provenance stamping shared by both cache implementations."""

    def _init_provenance(self, conv_id: str = "conv") -> None:
        self.conv_id = conv_id
        self.provenance: list[KVProvenance] = []
        self._turn_idx: int = 0
        self._agent_id: str = "a0"
        self._channel: str = CH_AGENT
        self._token_ids: list[int] = []

    def set_write_context(self, turn_idx: int, agent_id: str, channel: str) -> None:
        """Set the stamp applied to subsequent writes."""
        self._turn_idx = turn_idx
        self._agent_id = agent_id
        self._channel = channel

    def stamp(self, token_ids, start_position: int) -> None:
        """Record provenance for ``len(token_ids)`` newly written positions."""
        for i, tok in enumerate(token_ids):
            self.provenance.append(
                KVProvenance(
                    position=start_position + i,
                    turn_idx=self._turn_idx,
                    agent_id=self._agent_id,
                    channel=self._channel,
                    token_id=int(tok),
                )
            )
            self._token_ids.append(int(tok))

    @property
    def token_ids(self) -> list[int]:
        return list(self._token_ids)

    def provenance_frame(self) -> list[dict[str, Any]]:
        return [p.as_row() for p in self.provenance]


class TracingCache(ProvenanceMixin):
    """Self-contained KV cache with provenance stamping.

    Storage layout per layer: ``keys[l]`` and ``values[l]`` are tensors of shape
    ``[n_kv_heads, T, head_dim]``.  Batch size is fixed at one -- the design is
    single-conversation by construction, since provenance is only meaningful
    within one conversation.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        conv_id: str = "conv",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers
        self._init_provenance(conv_id)

    # -- HF-compatible surface ------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``key_states`` / ``value_states`` and return the full cache.

        Signature matches ``transformers.DynamicCache.update`` so the same call
        site works on both back-ends.  Incoming tensors are
        ``[n_kv_heads, n_new, head_dim]``.
        """
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = key_states.detach().clone()
            self._values[layer_idx] = value_states.detach().clone()
        else:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states.detach()], dim=1)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states.detach()], dim=1)
        return self._keys[layer_idx], self._values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        k = self._keys[layer_idx]
        return 0 if k is None else int(k.shape[1])

    def __len__(self) -> int:
        return self.get_seq_length(0)

    # -- accessors -------------------------------------------------------
    def keys(self, layer_idx: int) -> torch.Tensor:
        k = self._keys[layer_idx]
        if k is None:
            raise IndexError(f"layer {layer_idx} has no cached keys yet")
        return k

    def values(self, layer_idx: int) -> torch.Tensor:
        v = self._values[layer_idx]
        if v is None:
            raise IndexError(f"layer {layer_idx} has no cached values yet")
        return v

    def value_array(self) -> np.ndarray:
        """All value vectors as ``[n_layers, n_kv_heads, T, head_dim]``.

        This is what the Module 2 reservoir probes are fitted on.
        """
        return np.stack([self.values(l).cpu().numpy() for l in range(self.n_layers)], axis=0)

    def key_norms(self) -> np.ndarray:
        """L2 norm of every key, ``[n_layers, n_kv_heads, T]`` (a cheap statistic
        retained for *every* run, unlike the raw tensors)."""
        return np.stack(
            [self.keys(l).norm(dim=-1).cpu().numpy() for l in range(self.n_layers)], axis=0
        )

    def value_norms(self) -> np.ndarray:
        return np.stack(
            [self.values(l).norm(dim=-1).cpu().numpy() for l in range(self.n_layers)], axis=0
        )

    # -- multi-agent operations -----------------------------------------
    def clone(self, conv_id: str | None = None) -> "TracingCache":
        """Deep copy, used to fork a baseline cache for a seed-matched replay."""
        out = TracingCache(self.n_layers, self.n_kv_heads, self.head_dim,
                           conv_id or self.conv_id, self.dtype)
        out._keys = [None if k is None else k.clone() for k in self._keys]
        out._values = [None if v is None else v.clone() for v in self._values]
        out.provenance = list(self.provenance)
        out._token_ids = list(self._token_ids)
        return out

    def graft(self, peer: "TracingCache", positions: list[int], adapter) -> int:
        """Transplant peer cache entries -- the shared-KV memory arm.

        The rotary phase of a key encodes its absolute position, so every
        transplanted key is re-indexed from its source position to the position
        it lands on here.  Values carry no positional phase and are copied
        unchanged.  Shared-KV transfer is only defined between agents
        instantiating the *same* base model; that restriction is enforced by the
        caller.

        Returns the number of grafted positions.
        """
        if not positions:
            return 0
        base = self.get_seq_length(0)
        new_positions = list(range(base, base + len(positions)))
        for layer in range(self.n_layers):
            pk = peer.keys(layer)[:, positions, :]                       # [Hkv, n, dh]
            pv = peer.values(layer)[:, positions, :]
            rotated = adapter.rope_reindex(pk, positions, new_positions)
            self.update(rotated, pv, layer)
        prev_channel, prev_agent = self._channel, self._agent_id
        self._channel = CH_PEER
        for i, src in enumerate(positions):
            src_prov = peer.provenance[src]
            self.provenance.append(
                KVProvenance(
                    position=new_positions[i],
                    turn_idx=self._turn_idx,
                    agent_id=self._agent_id,
                    channel=CH_PEER,
                    token_id=src_prov.token_id,
                    source_position=src,
                )
            )
            self._token_ids.append(src_prov.token_id)
        self._channel, self._agent_id = prev_channel, prev_agent
        return len(positions)

    # -- diagnostics -----------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "convId": self.conv_id,
            "length": self.get_seq_length(0),
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "bytes_per_token": 2 * self.n_layers * self.n_kv_heads * self.head_dim * 4,
        }
