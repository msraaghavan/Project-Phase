"""Module 1: TracingCache provenance, RoPE re-indexing, replay determinism,
intervention equivalence, and the Hugging Face adapter on a random tiny Llama."""

import numpy as np
import pytest
import torch

from rcm.adapters import TinyAdapter, build_adapter
from rcm.adapters.base import InterventionSpec
from rcm.adapters.tiny import rope_frequencies, rope_rotate
from rcm.config import ARM_SHARED, ExperimentConfig
from rcm.module1.orchestrator import ConversationOrchestrator
from rcm.probes import BOS, encode


@pytest.fixture(scope="module")
def adapter():
    cfg = ExperimentConfig()
    return build_adapter(cfg)


def test_cache_stamps_provenance(adapter):
    cache = adapter.new_cache("c")
    cache.set_write_context(turn_idx=2, agent_id="a1", channel="probe")
    ids = [BOS] + encode("the g0a person is neg0 .")
    adapter.forward(ids, cache)
    assert cache.get_seq_length(0) == len(ids)
    assert len(cache.provenance) == len(ids)
    p = cache.provenance[3]
    assert (p.position, p.turn_idx, p.agent_id, p.channel, p.token_id) == (3, 2, "a1", "probe", ids[3])


def test_rope_reindex_is_exact(adapter):
    """Rotating a key by (new - old) equals encoding it at `new` directly."""
    inv = rope_frequencies(16, 10000.0)
    raw = torch.randn(2, 5, 16)
    old = np.arange(10, 15)
    new = np.arange(40, 45)
    k_old = rope_rotate(raw, torch.as_tensor(old, dtype=torch.float32)[:, None] * inv[None])
    k_new = rope_rotate(raw, torch.as_tensor(new, dtype=torch.float32)[:, None] * inv[None])
    assert torch.allclose(adapter.rope_reindex(k_old, old, new), k_new, atol=1e-5)


def test_graft_matches_native_encoding(adapter):
    """A grafted segment must land exactly where native writing would put it.

    Layer-0 keys depend only on the token and its RoPE phase, so after
    re-indexing they must equal a native write bit-for-bit.  Deeper layers
    legitimately differ because their keys encode the *sender's* context --
    that non-portability is the very reason shared-KV transfer is restricted
    to agents on the same base model.
    """
    ids_a = [BOS] + encode("they said one and one met there .")
    ids_b = encode("the g0a person is neg0 .")
    native = adapter.new_cache("n")
    adapter.forward(ids_a, native)
    adapter.forward(ids_b, native)              # written natively at positions 9..
    peer = adapter.new_cache("p")
    adapter.forward([BOS] + encode("q0 answer a0 ."), peer)
    # peer writes ids_b at *different* positions, then we graft into a fresh receiver
    start = peer.get_seq_length(0)
    adapter.forward(ids_b, peer)
    recv = adapter.new_cache("r")
    adapter.forward(ids_a, recv)
    recv.graft(peer, list(range(start, start + len(ids_b))), adapter)
    assert torch.allclose(recv.keys(0)[:, len(ids_a):], native.keys(0)[:, len(ids_a):], atol=1e-5)
    assert torch.allclose(recv.values(0)[:, len(ids_a):], native.values(0)[:, len(ids_a):], atol=1e-6)
    assert recv.get_seq_length(0) == native.get_seq_length(0)
    assert recv.provenance[-1].channel == "peer"
    assert recv.provenance[-1].source_position == start + len(ids_b) - 1


def test_replay_is_deterministic(adapter):
    cfg = ExperimentConfig()
    cfg.run.n_turns = 4
    orch = ConversationOrchestrator(adapter, cfg)
    a = orch.run(ARM_SHARED, 0.5, 3)
    b = orch.run(ARM_SHARED, 0.5, 3)
    assert a.token_ids() == b.token_ids()
    assert a.final_bias == b.final_bias
    assert np.allclose(a.stats["alpha_bar"], b.stats["alpha_bar"])


def test_gate_equals_logit_bias_and_ablate_removes_mass(adapter):
    """gate(g) post-softmax == adding log g pre-softmax; ablate zeroes the targets."""
    ids = [BOS] + encode("the g1a person is neg1 . the g1b person is pos1 .")
    targets = {(0, 0): [2, 3], (2, 1): [5]}
    tap0 = adapter.new_tap()
    adapter.forward(ids, adapter.new_cache("clean"), tap=tap0)
    clean = tap0.finalize()["mass"]
    cache = adapter.new_cache("x")
    tap = adapter.new_tap()
    adapter.forward(ids, cache, InterventionSpec("gate", targets, gate=0.2), tap=tap)
    gated = tap.finalize()["mass"]
    cache2 = adapter.new_cache("y")
    tap2 = adapter.new_tap()
    adapter.forward(ids, cache2, InterventionSpec("ablate", targets), tap=tap2)
    abl = tap2.finalize()["mass"]
    # ablated heads receive exactly zero mass at target positions, others untouched
    for h in adapter.query_heads_of(0):
        assert abl[0, h, 2] == 0 and abl[0, h, 3] == 0
    for h in adapter.query_heads_of(1):
        assert abl[2, h, 5] == 0
    # gating lowers the targeted heads' mass on the targets; layer 0's other
    # kv-head sees identical inputs, so its attention must not move at all
    for h in adapter.query_heads_of(0):
        assert 0 < gated[0, h, 2] < clean[0, h, 2] and 0 < gated[0, h, 3] < clean[0, h, 3]
    for h in adapter.query_heads_of(1):
        assert np.allclose(gated[0, h], clean[0, h]) and np.allclose(abl[0, h], clean[0, h])
    # logit-bias formulation reproduces gating row by row
    q, k = torch.randn(3, 8), torch.randn(3, 8)
    s = q @ k.T
    a = torch.softmax(s, -1)
    a[:, 1] *= 0.2
    a = a / a.sum(-1, keepdim=True)
    s2 = s.clone()
    s2[:, 1] += np.log(0.2)
    assert torch.allclose(a, torch.softmax(s2, -1), atol=1e-6)


def test_perturb_removes_direction_at_read_time(adapter):
    """The edit projects the bias direction out of the targeted value vectors only,
    and only inside the forward pass that carries it: the stored cache is untouched."""
    ids = [BOS] + encode("the g2a person is neg2 .")
    direction = np.random.default_rng(0).normal(size=adapter.head_dim)
    w_hat = torch.as_tensor(direction / np.linalg.norm(direction), dtype=torch.float32)
    spec = InterventionSpec("perturb", {(1, 0): [3]}, {(1, 0): direction}, epsilon=1.0)
    assert spec.is_active() and spec.n_entries() == 1
    cache = adapter.new_cache("c")
    adapter.forward(ids, cache)
    v = cache.values(1)                                           # [kv_heads, T, head_dim]
    v_eff = adapter.model.blocks[1].attn._perturb_values(v, spec)
    assert abs(float(v_eff[0, 3] @ w_hat)) < 1e-5                 # direction removed ...
    assert torch.allclose(v_eff[0, 3], v[0, 3] - (v[0, 3] @ w_hat) * w_hat, atol=1e-6)
    untouched = torch.ones(v.shape[:2], dtype=torch.bool)
    untouched[0, 3] = False
    assert torch.equal(v_eff[untouched], v[untouched])            # ... and nothing else moves
    before = cache.values(1).clone()
    plain = adapter.next_token_logits(encode("they said"), cache)
    edited = adapter.next_token_logits(encode("they said"), cache, spec)
    assert torch.equal(cache.values(1), before)                   # read-time only
    assert not np.allclose(plain, edited)


def test_hf_adapter_on_random_llama():
    """The production path: HFTracingCache + eager attention + hook interventions."""
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from rcm.adapters.hf import HFAdapter, HFTracingCache

    lcfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                       attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(lcfg).eval()
    cfg = ExperimentConfig()
    cfg.model.backend = "hf"
    ad = HFAdapter(cfg, model=model)
    assert (ad.n_layers, ad.n_heads, ad.n_kv_heads, ad.head_dim) == (2, 4, 2, 16)

    cache = ad.new_cache("hf")
    assert isinstance(cache, HFTracingCache) and isinstance(cache, transformers.DynamicCache)
    tap = ad.new_tap()
    cache.set_write_context(1, "a0", "probe")
    ad.forward([1, 2, 3, 4, 5], cache, tap=tap)
    assert cache.get_seq_length() == 5 and len(cache.provenance) == 5
    assert cache.provenance[2].channel == "probe" and cache.provenance[2].token_id == 3
    st = tap.finalize()
    assert st["seq_len"] == 5 and np.allclose(st["alpha_bar"].sum(1), 1.0, atol=1e-5)

    # ablation through the forward pre-hook zeroes attention on the target head only
    tap2 = ad.new_tap()
    c2 = ad.new_cache("hf2")
    spec = InterventionSpec("ablate", {(0, 0): [1]})
    ad.forward([1, 2, 3, 4, 5], c2, spec, tap=tap2)
    mass = tap2.finalize()["mass"]
    assert mass[0, 0, 1] == 0 and mass[0, 1, 1] == 0        # query heads 0,1 read kv-head 0
    assert mass[0, 2, 1] > 0 and mass[1, 0, 1] > 0           # other heads / layers untouched

    out = ad.generate([1, 2, 3], ad.new_cache("g"), max_new_tokens=3)
    assert len(out.token_ids) == 3
