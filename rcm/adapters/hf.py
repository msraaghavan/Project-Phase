"""Hugging Face adapter -- the production path of the design document.

This is the adapter used against the frozen open-weight models named in the
design (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B).  Nothing in
Modules 2-4 changes when it is selected; only this file knows about
``transformers``.

Three things make the HF path work:

1. :class:`HFTracingCache` subclasses ``transformers.DynamicCache`` and
   intercepts every ``update`` so each write is provenance-stamped, while the
   tensors themselves stay in the stock cache.
2. Attention is requested with ``output_attentions=True`` on a model loaded with
   ``attn_implementation="eager"``, because FlashAttention and SDPA never
   materialise ``alpha^{l,h}_{q,t}``.
3. Ablation and gating are injected by a forward pre-hook on each attention
   module that rewrites its ``attention_mask`` into a per-(layer, head) additive
   bias: ``-inf`` for ablation, ``log g`` for gating.  Adding ``log g`` before
   the softmax is *exactly* "scale the target attention by g and renormalise",
   so one mechanism implements both modes, and the HF and reference back-ends
   are provably equivalent (``tests/test_interventions.py``).

The module imports ``transformers`` lazily, so the rest of the package runs
without it installed.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcm.adapters.base import InterventionSpec, ModelAdapter, TurnOutput
from rcm.module1.cache import ProvenanceMixin
from rcm.module1.hooks import AttentionTap
from rcm.probes import SURFACE, render

try:  # pragma: no cover - exercised only when transformers is installed
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    _HF_AVAILABLE = True
except Exception:  # pragma: no cover
    DynamicCache = object  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False


class HFTracingCache(DynamicCache, ProvenanceMixin):  # type: ignore[misc,valid-type]
    """``DynamicCache`` with provenance stamping on every write."""

    def __init__(self, *args, conv_id: str = "conv", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_provenance(conv_id)
        self._pending_tokens: list[int] | None = None

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if layer_idx == 0:
            # provenance is per position, so stamp once per token, not per layer
            n_new = int(key_states.shape[-2])
            start = int(keys.shape[-2]) - n_new
            if self._pending_tokens is not None and len(self._pending_tokens) == n_new:
                self.stamp(self._pending_tokens, start)
                self._pending_tokens = None
        return keys, values

    def set_pending_tokens(self, token_ids) -> None:
        """Tokens whose KV entries the next forward pass will write."""
        self._pending_tokens = [int(t) for t in token_ids]

    def layer_keys(self, layer_idx: int) -> torch.Tensor:
        return self.layers[layer_idx].keys[0]      # [n_kv_heads, T, head_dim]

    def layer_values(self, layer_idx: int) -> torch.Tensor:
        return self.layers[layer_idx].values[0]

    def value_array(self) -> np.ndarray:
        return np.stack(
            [self.layer_values(i).float().cpu().numpy() for i in range(len(self.layers))], axis=0
        )

    def key_norms(self) -> np.ndarray:
        return np.stack([self.layer_keys(i).float().norm(dim=-1).cpu().numpy()
                         for i in range(len(self.layers))], axis=0)

    def value_norms(self) -> np.ndarray:
        return np.stack([self.layer_values(i).float().norm(dim=-1).cpu().numpy()
                         for i in range(len(self.layers))], axis=0)

    def clone(self, conv_id: str | None = None) -> "HFTracingCache":
        """Deep copy, so scoring and replays never touch the live memory."""
        out = copy.deepcopy(self)
        if conv_id:
            out.conv_id = conv_id
        return out

    def graft(self, peer: "HFTracingCache", positions: list[int], adapter) -> int:
        """Transplant peer entries with RoPE re-indexing (shared-KV arm)."""
        if not positions:
            return 0
        base = self.get_seq_length()
        new_positions = list(range(base, base + len(positions)))
        self._pending_tokens = None
        for layer in range(len(peer.layers)):
            pk = peer.layers[layer].keys[:, :, positions, :]        # [1, Hkv, n, dh]
            pv = peer.layers[layer].values[:, :, positions, :]
            rotated = adapter.rope_reindex(pk, positions, new_positions)
            super().update(rotated.contiguous(), pv.clone(), layer)
        self._stamp_peer(peer, positions, new_positions)
        return len(positions)


class _AttentionMaskInjector:
    """Forward pre-hook that rewrites one attention module's mask.

    Ablation and gating are both additive biases on the pre-softmax logits, so a
    single hook implements both.  The hook is attached only for the duration of
    an intervened replay and removed afterwards.
    """

    def __init__(self, adapter: "HFAdapter", spec: InterventionSpec, layer_idx: int) -> None:
        self.adapter = adapter
        self.spec = spec
        self.layer_idx = layer_idx

    def __call__(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            return args, kwargs
        q_len = int(hidden.shape[1])
        pos_start = int(self.adapter._pos_start)
        k_len = pos_start + q_len
        device, dtype = hidden.device, hidden.dtype

        q_abs = torch.arange(pos_start, pos_start + q_len, device=device).view(q_len, 1)
        k_abs = torch.arange(k_len, device=device).view(1, k_len)
        causal = torch.where(k_abs <= q_abs, 0.0, float("-inf")).to(dtype)
        mask = causal.view(1, 1, q_len, k_len).expand(
            1, self.adapter.n_heads, q_len, k_len
        ).clone()

        bias = float("-inf") if self.spec.mode == "ablate" else math.log(max(self.spec.gate, 1e-12))
        for (layer, kv_head), positions in self.spec.targets.items():
            if layer != self.layer_idx:
                continue
            idx = [p for p in positions if p < k_len]
            if not idx:
                continue
            for h in self.adapter.query_heads_of(kv_head):
                mask[0, h, :, idx] = bias
        kwargs["attention_mask"] = mask
        return args, kwargs


class HFAdapter(ModelAdapter):
    """Adapter over any Hugging Face decoder-only causal LM."""

    def __init__(self, cfg, model=None, tokenizer=None) -> None:
        if not _HF_AVAILABLE:
            raise ImportError("transformers is required for the 'hf' backend")
        self.cfg = cfg
        model_id = cfg.model.hf_model_id
        if model is None:
            if not model_id:
                raise ValueError("model.hf_model_id must be set for the 'hf' backend")
            device, dtype = _device_and_dtype()
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager",
            ).to(device)
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = model.eval()
        self.tokenizer = tokenizer
        for p in self.model.parameters():          # the model under study is frozen
            p.requires_grad_(False)
        hc = self.model.config
        self._n_layers = int(hc.num_hidden_layers)
        self._n_heads = int(hc.num_attention_heads)
        self._n_kv_heads = int(getattr(hc, "num_key_value_heads", hc.num_attention_heads))
        self._head_dim = int(getattr(hc, "head_dim", hc.hidden_size // hc.num_attention_heads))
        rope = getattr(hc, "rope_parameters", None) or {}
        self._rope_theta = float(getattr(hc, "rope_theta", None) or rope.get("rope_theta") or 10000.0)
        tok = self.tokenizer
        self._special_ids = sorted(set(tok.all_special_ids)) if tok is not None else []
        start = None if tok is None else (tok.bos_token_id if tok.bos_token_id is not None
                                          else tok.eos_token_id)
        self._bos = [] if start is None else [int(start)]
        self._pos_start = 0

    # -- geometry --------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def n_kv_heads(self) -> int:
        return self._n_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def name(self) -> str:
        return self.cfg.model.hf_model_id or self.model.config.model_type

    def describe(self) -> dict:
        p = next(self.model.parameters())
        return super().describe() | {"device": str(p.device), "dtype": str(p.dtype).replace("torch.", "")}

    # -- text codec: the probes rendered in the model's own vocabulary ------
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(" " + render(text), add_special_tokens=False)

    def bos_ids(self) -> list[int]:
        return list(self._bos)

    def word_ids(self, word: str) -> list[int]:
        return self.tokenizer.encode(" " + SURFACE.get(word, word), add_special_tokens=False)

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)])

    # -- execution -------------------------------------------------------
    def new_cache(self, conv_id: str = "conv") -> HFTracingCache:
        return HFTracingCache(config=self.model.config, conv_id=conv_id)

    def new_tap(self, top_k: int = 8) -> AttentionTap:
        return AttentionTap(self.n_layers, self.n_heads, self.cfg.model.max_seq_len, top_k)

    def _attention_modules(self):
        return [layer.self_attn for layer in self.model.model.layers]

    def _install_hooks(self, spec: InterventionSpec | None):
        if spec is None or spec.mode not in ("ablate", "gate") or not spec.targets:
            return []
        return [
            mod.register_forward_pre_hook(
                _AttentionMaskInjector(self, spec, i), with_kwargs=True
            )
            for i, mod in enumerate(self._attention_modules())
        ]

    @torch.no_grad()
    def forward(self, token_ids, cache=None, intervention=None, capture=True, tap=None):
        ids = torch.as_tensor(token_ids, dtype=torch.long).view(1, -1).to(self.model.device)
        self._pos_start = cache.get_seq_length() if cache is not None else 0
        if cache is not None:
            cache.set_pending_tokens(ids[0].tolist())
        undo = []
        if intervention is not None and intervention.mode == "perturb" and cache is not None:
            undo = apply_perturbation_inplace(cache, intervention, self.n_layers)
        handles = self._install_hooks(intervention)
        try:
            out = self.model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=cache is not None,
                output_attentions=bool(capture and tap is not None),
            )
        finally:
            for h in handles:
                h.remove()
            for layer, kv_head, idx, original in undo:     # the edit is read-time only
                cache.layers[layer].values[0][kv_head, idx, :] = original
        if capture and tap is not None:
            attns = torch.stack([a[0] for a in out.attentions]).float().cpu()  # one transfer
            for layer_idx in range(attns.shape[0]):
                tap.observe(layer_idx, attns[layer_idx], self._pos_start)
        return out.logits[0]

    @torch.no_grad()
    def generate(self, prompt_ids, cache, max_new_tokens, intervention=None,
                 tap=None, forced_ids=None) -> TurnOutput:
        prompt_ids = [int(t) for t in np.asarray(prompt_ids).reshape(-1)]
        logits = self.forward(prompt_ids, cache, intervention, capture=True, tap=tap)
        generated: list[int] = []
        step_logits: list[np.ndarray] = []
        for step in range(max_new_tokens):
            row = logits[-1].float().clone()
            row[self._special_ids] = float("-inf")       # a turn never ends the document
            step_logits.append(row.cpu().numpy())
            if forced_ids is not None:
                if step >= len(forced_ids):
                    break
                nxt = int(forced_ids[step])
            else:
                nxt = int(torch.argmax(row).item())
            generated.append(nxt)
            logits = self.forward([nxt], cache, intervention, capture=True, tap=tap)
        return TurnOutput(
            token_ids=generated,
            text=self.tokenizer.decode(generated) if self.tokenizer else str(generated),
            logits=np.stack(step_logits) if step_logits else np.zeros((0, 1)),
            n_prompt_tokens=len(prompt_ids),
        )

    @torch.no_grad()
    def next_token_logits(self, prompt_ids, cache=None, intervention=None) -> np.ndarray:
        """Logits after ``prompt_ids``, conditioned on a *copy* of ``cache``."""
        work = cache.clone("probe") if cache is not None else None
        logits = self.forward(list(prompt_ids), work, intervention, capture=False)
        return logits[-1].float().cpu().numpy()

    @torch.no_grad()
    def continuation_logprob(self, prompt_ids, cont_ids, cache=None, intervention=None) -> float:
        work = cache.clone("probe") if cache is not None else None
        prompt_ids, cont_ids = list(prompt_ids), list(cont_ids)
        logits = self.forward(prompt_ids + cont_ids, work, intervention, capture=False)
        logp = F.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        return float(sum(logp[start + i, t].item() for i, t in enumerate(cont_ids)))

    @torch.no_grad()
    def sequence_logprob(self, token_ids, intervention=None) -> float:
        ids = torch.as_tensor(token_ids, dtype=torch.long).view(1, -1).to(self.model.device)
        if ids.shape[1] < 2:
            return 0.0
        out = self.model(input_ids=ids[:, :-1], use_cache=False)
        logp = F.log_softmax(out.logits[0].float(), dim=-1)
        tgt = ids[0, 1:]
        return float(logp[torch.arange(len(tgt)), tgt].sum().item())

    def rope_reindex(self, keys: torch.Tensor, old_pos, new_pos) -> torch.Tensor:
        """Rotate RoPE-encoded keys by the position delta.

        Hugging Face uses the half-rotation (``rotate_half``) layout rather than
        the interleaved one, so the pairing differs from the reference model --
        exactly the kind of model-specific detail the adapter exists to hide.
        """
        head_dim = keys.shape[-1]
        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=keys.device)
        inv = 1.0 / (self._rope_theta ** (i / head_dim))
        delta = torch.as_tensor(
            np.asarray(new_pos) - np.asarray(old_pos), dtype=torch.float32, device=keys.device
        )
        ang = delta.unsqueeze(-1) * inv.unsqueeze(0)
        cos, sin = torch.cos(ang).to(keys.dtype), torch.sin(ang).to(keys.dtype)
        half = head_dim // 2
        x1, x2 = keys[..., :half], keys[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_perturbation_inplace(cache, spec: InterventionSpec, n_layers: int) -> list:
    """Project the bias direction out of the target value vectors.

    This is the literal reading of "perturbation projects the probe-identified
    bias direction out of the target value vectors".  Returns the overwritten
    slices so :meth:`HFAdapter.forward` can restore them after the pass, which
    makes the edit read-time only, exactly as in the reference adapter.
    """
    undo = []
    for (layer, kv_head), positions in spec.targets.items():
        if layer >= n_layers:
            continue
        w = spec.directions.get((layer, kv_head))
        if w is None or not positions:
            continue
        values = cache.layers[layer].values[0]              # [n_kv_heads, T, head_dim]
        idx = [p for p in positions if p < values.shape[1]]
        if not idx:
            continue
        w_t = torch.as_tensor(np.asarray(w), dtype=values.dtype, device=values.device)
        w_hat = w_t / w_t.norm().clamp_min(1e-12)
        sel = values[kv_head, idx, :]
        undo.append((layer, kv_head, idx, sel.clone()))
        proj = (sel @ w_hat).unsqueeze(-1) * w_hat.unsqueeze(0)
        values[kv_head, idx, :] = sel - spec.epsilon * proj
    return undo


def _device_and_dtype():
    """GPU when available (bf16 on Ampere+, fp16 before); float32 on CPU.

    ``RCM_DEVICE`` and ``RCM_DTYPE`` override the choice, e.g. ``RCM_DTYPE=float32``
    for a small model on a pre-Ampere card.
    """
    device = os.environ.get("RCM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("RCM_DTYPE"):
        dtype = getattr(torch, os.environ["RCM_DTYPE"])
    elif device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype
