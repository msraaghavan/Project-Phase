"""Reference decoder-only transformer and its adapter.

This is a *real* transformer, not a mock: causal self-attention with grouped
query attention (GQA), rotary position embeddings (RoPE), RMSNorm and residual
MLP blocks -- the same structural family as Llama-3, Mistral and Qwen2.5.  It is
simply small enough (4 layers, 8 query heads, 2 KV heads, d=128) to train and
run on a CPU in seconds, which is what makes the whole pipeline reproducible on
a laptop.

Attention is computed in eager mode so the per-head distribution
``alpha^{l,h}_{q,t}`` is materialised and can be tapped; fused kernels such as
FlashAttention and SDPA never expose it.  That is the same constraint the
Hugging Face path lives under.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcm.adapters.base import InterventionSpec, ModelAdapter, TurnOutput
from rcm.config import ModelConfig
from rcm.module1.cache import TracingCache
from rcm.module1.hooks import AttentionTap
from rcm.probes import EOS, PAD, VOCAB_SIZE, decode


# --------------------------------------------------------------------------
# Rotary position embeddings
# --------------------------------------------------------------------------
def rope_frequencies(head_dim: int, theta: float, device=None) -> torch.Tensor:
    """Inverse frequencies ``theta^(-2i/d)`` for ``i = 0 .. d/2-1``."""
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (theta ** (i / head_dim))


def rope_rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[..., T, head_dim]`` by ``angles`` ``[T, head_dim/2]``.

    Because a rotation by ``p`` followed by a rotation by ``d`` equals a
    rotation by ``p + d``, the same routine both applies RoPE at write time and
    *re-indexes* an already-rotated key at transplant time.
    """
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------
class TinyAttention(nn.Module):
    """Causal GQA attention with RoPE, eager softmax and intervention points."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.register_buffer("inv_freq", rope_frequencies(cfg.head_dim, cfg.rope_theta),
                             persistent=False)

    def _angles(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,                     # [T, d_model]
        cache: TracingCache | None,
        pos_start: int,
        tap: AttentionTap | None = None,
        intervention: InterventionSpec | None = None,
    ) -> torch.Tensor:
        T = x.shape[0]
        H, Hkv, dh = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(T, H, dh).transpose(0, 1)        # [H, T, dh]
        k = self.k_proj(x).view(T, Hkv, dh).transpose(0, 1)      # [Hkv, T, dh]
        v = self.v_proj(x).view(T, Hkv, dh).transpose(0, 1)

        positions = torch.arange(pos_start, pos_start + T, device=x.device)
        ang = self._angles(positions)                            # [T, dh/2]
        q = rope_rotate(q, ang)
        k = rope_rotate(k, ang)

        if cache is not None:
            k_all, v_all = cache.update(k, v, self.layer_idx)
        else:
            k_all, v_all = k, v
        S = k_all.shape[1]

        # -- Module 4 intervention: perturbation acts on value vectors -----
        v_eff = v_all
        if intervention is not None and intervention.mode == "perturb":
            v_eff = self._perturb_values(v_all, intervention)

        kq = k_all.repeat_interleave(self.group, dim=0)          # [H, S, dh]
        vq = v_eff.repeat_interleave(self.group, dim=0)

        scores = torch.matmul(q, kq.transpose(-1, -2)) / math.sqrt(dh)   # [H, T, S]

        key_pos = torch.arange(S, device=x.device).view(1, 1, S)
        causal = key_pos <= positions.view(1, T, 1)
        scores = scores.masked_fill(~causal, float("-inf"))

        # -- Module 4 intervention: ablation acts pre-softmax --------------
        if intervention is not None and intervention.mode == "ablate":
            scores = self._ablate_scores(scores, intervention, S)

        attn = torch.softmax(scores, dim=-1)

        # -- Module 4 intervention: gating acts post-softmax, then renormalise
        if intervention is not None and intervention.mode == "gate":
            attn = self._gate_attention(attn, intervention, S)

        if tap is not None:
            tap.observe(self.layer_idx, attn, pos_start)

        out = torch.matmul(attn, vq)                             # [H, T, dh]
        out = out.transpose(0, 1).reshape(T, H * dh)
        return self.o_proj(out)

    def forward_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Batched full-sequence attention, used only for testbed training.

        Mathematically identical to :meth:`forward` with no cache and no
        intervention -- ``tests/test_adapter.py`` asserts the two agree to
        floating-point tolerance -- but it processes a whole batch at once,
        which is what makes CPU training take seconds rather than minutes.
        """
        B, T, _ = x.shape
        H, Hkv, dh = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)      # [B, H, T, dh]
        k = self.k_proj(x).view(B, T, Hkv, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, dh).transpose(1, 2)

        ang = self._angles(torch.arange(T, device=x.device))      # [T, dh/2]
        q, k = rope_rotate(q, ang), rope_rotate(k, ang)

        kq = k.repeat_interleave(self.group, dim=1)
        vq = v.repeat_interleave(self.group, dim=1)
        scores = torch.matmul(q, kq.transpose(-1, -2)) / math.sqrt(dh)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, vq).transpose(1, 2).reshape(B, T, H * dh)
        return self.o_proj(out)

    # -- intervention primitives ----------------------------------------
    def _target_index(self, spec: InterventionSpec, kv_head: int, S: int, device):
        pos = [p for p in spec.positions(self.layer_idx, kv_head) if p < S]
        if not pos:
            return None
        return torch.as_tensor(pos, dtype=torch.long, device=device)

    def _ablate_scores(self, scores: torch.Tensor, spec: InterventionSpec, S: int) -> torch.Tensor:
        scores = scores.clone()
        for kv_head in range(self.n_kv_heads):
            idx = self._target_index(spec, kv_head, S, scores.device)
            if idx is None:
                continue
            for h in range(kv_head * self.group, (kv_head + 1) * self.group):
                scores[h][:, idx] = float("-inf")
        return scores

    def _gate_attention(self, attn: torch.Tensor, spec: InterventionSpec, S: int) -> torch.Tensor:
        attn = attn.clone()
        for kv_head in range(self.n_kv_heads):
            idx = self._target_index(spec, kv_head, S, attn.device)
            if idx is None:
                continue
            for h in range(kv_head * self.group, (kv_head + 1) * self.group):
                attn[h][:, idx] = attn[h][:, idx] * spec.gate
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _perturb_values(self, v_all: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        v_eff = v_all.clone()
        S = v_all.shape[1]
        for kv_head in range(self.n_kv_heads):
            idx = self._target_index(spec, kv_head, S, v_all.device)
            if idx is None:
                continue
            w = spec.directions.get((self.layer_idx, kv_head))
            if w is None:
                continue
            w_t = torch.as_tensor(np.asarray(w), dtype=v_all.dtype, device=v_all.device)
            w_hat = w_t / w_t.norm().clamp_min(1e-12)
            sel = v_eff[kv_head, idx, :]                          # [n, dh]
            proj = (sel @ w_hat).unsqueeze(-1) * w_hat.unsqueeze(0)
            v_eff[kv_head, idx, :] = sel - spec.epsilon * proj
        return v_eff


class TinyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = TinyAttention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )

    def forward(self, x, cache, pos_start, tap=None, intervention=None):
        x = x + self.attn(self.attn_norm(x), cache, pos_start, tap, intervention)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward_batched(self, x):
        x = x + self.attn.forward_batched(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyTransformer(nn.Module):
    """Decoder-only transformer over the closed probe vocabulary."""

    def __init__(self, cfg: ModelConfig, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, token_ids, cache=None, pos_start=0, tap=None, intervention=None):
        """Incremental forward over a 1-D token sequence -> ``[T, vocab]``."""
        ids = torch.as_tensor(token_ids, dtype=torch.long).view(-1)
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, cache, pos_start, tap, intervention)
        return self.lm_head(self.norm(x))

    def forward_batch(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence batched forward over ``[B, T]`` -> ``[B, T, vocab]``."""
        ids = torch.as_tensor(batch_ids, dtype=torch.long)
        x = self.embed(ids)
        for block in self.blocks:
            x = block.forward_batched(x)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------
class TinyAdapter(ModelAdapter):
    """:class:`ModelAdapter` over :class:`TinyTransformer`."""

    def __init__(self, cfg, model: TinyTransformer | None = None) -> None:
        self.cfg = cfg
        self.mcfg: ModelConfig = cfg.model
        self.model = model or TinyTransformer(self.mcfg)
        self.model.eval()
        for p in self.model.parameters():          # the model under study is frozen
            p.requires_grad_(False)
        self._inv_freq = rope_frequencies(self.mcfg.head_dim, self.mcfg.rope_theta)

    # -- geometry --------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return self.mcfg.n_layers

    @property
    def n_heads(self) -> int:
        return self.mcfg.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.mcfg.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.mcfg.head_dim

    @property
    def name(self) -> str:
        return self.mcfg.name

    # -- execution -------------------------------------------------------
    def new_cache(self, conv_id: str = "conv") -> TracingCache:
        return TracingCache(self.n_layers, self.n_kv_heads, self.head_dim, conv_id)

    def new_tap(self, top_k: int = 8) -> AttentionTap:
        return AttentionTap(self.n_layers, self.n_heads, self.mcfg.max_seq_len, top_k)

    @torch.no_grad()
    def forward(self, token_ids, cache=None, intervention=None, capture=True, tap=None):
        pos_start = cache.get_seq_length(0) if cache is not None else 0
        if cache is not None:
            cache.stamp(list(np.asarray(token_ids).reshape(-1)), pos_start)
        return self.model(token_ids, cache, pos_start, tap if capture else None, intervention)

    @torch.no_grad()
    def generate(self, prompt_ids, cache, max_new_tokens, intervention=None,
                 tap=None, forced_ids=None) -> TurnOutput:
        """Greedy decoding.

        ``forced_ids`` replays a previously recorded turn token-for-token.  That
        is what makes a replay *seed-matched*: the intervened and baseline runs
        see identical token sequences, so any behavioural difference is
        attributable to the KV edit alone.
        """
        prompt_ids = list(np.asarray(prompt_ids).reshape(-1))
        logits = self.forward(prompt_ids, cache, intervention, capture=True, tap=tap)
        generated: list[int] = []
        step_logits: list[np.ndarray] = []
        for step in range(max_new_tokens):
            row = logits[-1].clone()
            row[[PAD, EOS]] = float("-inf")     # a turn never ends the document (min_new_tokens)
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
            text=decode(generated),
            logits=np.stack(step_logits) if step_logits else np.zeros((0, self.model.vocab_size)),
            n_prompt_tokens=len(prompt_ids),
        )

    @torch.no_grad()
    def next_token_logits(self, prompt_ids, cache=None, intervention=None) -> np.ndarray:
        """Logits for the token after ``prompt_ids``, conditioned on ``cache``.

        The cache is *cloned* so the measurement leaves no trace -- scoring a
        turn must not change the runtime memory it is scoring.
        """
        work = cache.clone("probe") if cache is not None else None
        logits = self.forward(prompt_ids, work, intervention, capture=False)
        return logits[-1].cpu().numpy()

    @torch.no_grad()
    def sequence_logprob(self, token_ids, intervention=None) -> float:
        ids = [int(t) for t in np.asarray(token_ids).reshape(-1) if int(t) != PAD]
        if len(ids) < 2:
            return 0.0
        logits = self.model(ids[:-1], None, 0, None, intervention)
        logp = F.log_softmax(logits, dim=-1)
        tgt = torch.as_tensor(ids[1:], dtype=torch.long)
        return float(logp[torch.arange(len(tgt)), tgt].sum().item())

    def rope_reindex(self, keys: torch.Tensor, old_pos, new_pos) -> torch.Tensor:
        """Rotate keys by the position delta (see :func:`rope_rotate`)."""
        delta = torch.as_tensor(np.asarray(new_pos) - np.asarray(old_pos), dtype=torch.float32)
        angles = delta.unsqueeze(-1) * self._inv_freq.unsqueeze(0)     # [n, dh/2]
        return rope_rotate(keys, angles)

    # -- testbed training -------------------------------------------------
    def train_on_corpus(self, corpus: np.ndarray, steps: int = 400, batch: int = 32,
                        lr: float = 3e-3, seed: int = 0, log_every: int = 100) -> list[float]:
        """Fit the reference transformer on the controlled corpus.

        This happens **once, before any experiment**, to give the testbed model
        a known bias structure.  During experiments the weights are frozen: no
        gradient is ever taken inside the measurement or intervention paths.
        """
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos"
        )
        data = torch.as_tensor(corpus, dtype=torch.long)
        losses: list[float] = []
        for step in range(steps):
            idx = rng.integers(0, len(data), size=batch)
            b = data[idx]
            logits = self.model.forward_batch(b[:, :-1])
            tgt = b[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size), tgt.reshape(-1),
                ignore_index=PAD,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            if log_every and (step + 1) % log_every == 0:
                print(f"    [train] step {step + 1:4d}/{steps}  loss {np.mean(losses[-log_every:]):.4f}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return losses

    def save(self, path) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "cfg": self.mcfg.__dict__}, path)

    def load(self, path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
