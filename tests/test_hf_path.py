"""The production (Hugging Face) path on a randomly initialised Qwen2 model.

A word-level stand-in tokenizer is used so no download is needed; long words
are split in two pieces so the multi-token scoring path is exercised too.
"""

import numpy as np
import pytest
import torch

from rcm.adapters.base import InterventionSpec
from rcm.config import ARM_CONTROL, ARM_SHARED, ARM_TEXT, ExperimentConfig
from rcm.module1.orchestrator import ConversationOrchestrator, raw_kv_selected
from rcm.module1.scoring import bias_index, hallucinated
from rcm.probes import BOS, STOI, ProbeBank, encode

transformers = pytest.importorskip("transformers")


class WordTokenizer:
    """Deterministic word-level tokenizer; words longer than 6 letters take two ids."""

    bos_token_id = None
    eos_token_id = 1
    all_special_ids = [0, 1]

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.itos: dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self.stoi: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.stoi:
            self.stoi[piece] = 2 + len(self.stoi)
            assert self.stoi[piece] < self.vocab_size
            self.itos[self.stoi[piece]] = piece
        return self.stoi[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        for mark in ".,?":
            text = text.replace(mark, " " + mark)
        ids = []
        for w in text.split():
            ids += [self._id(w[:4]), self._id("##" + w[4:])] if len(w) > 6 else [self._id(w)]
        return ids

    def decode(self, ids) -> str:
        return " ".join(self.itos.get(int(i), "?") for i in ids)


@pytest.fixture(scope="module")
def adapter():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from rcm.adapters.hf import HFAdapter

    qcfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                       rope_theta=1e6, attn_implementation="eager")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(qcfg).eval()
    cfg = ExperimentConfig()
    cfg.model.backend = "hf"
    cfg.model.max_seq_len = 1024
    return HFAdapter(cfg, model=model, tokenizer=WordTokenizer(256))


def _full_logits(ad, ids):
    return ad.model(input_ids=torch.as_tensor([ids]), use_cache=False).logits[0].float()


def test_rope_theta_read_from_rope_parameters(adapter):
    assert adapter._rope_theta == 1e6


def test_next_token_logits_conditions_on_cache_without_changing_it(adapter):
    ctx = adapter.bos_ids() + adapter.encode("the g0a person is neg0 . the g0a person is neg0 .")
    prompt = adapter.encode("the g0a person is")
    cache = adapter.new_cache("c")
    adapter.forward(ctx, cache)
    n = cache.get_seq_length()
    with_cache = adapter.next_token_logits(prompt, cache=cache)
    assert cache.get_seq_length() == n and len(cache.provenance) == n
    expected = _full_logits(adapter, ctx + prompt)[-1].numpy()
    assert np.allclose(with_cache, expected, atol=1e-4)
    without = adapter.next_token_logits(prompt)
    assert not np.allclose(with_cache, without, atol=1e-3)


def test_continuation_logprob_matches_full_forward(adapter):
    ctx = adapter.bos_ids() + adapter.encode("they said one and one met there .")
    cache = adapter.new_cache("c")
    adapter.forward(ctx, cache)
    prompt = adapter.encode("the g1a person is")
    cont = adapter.word_ids("pos1")                    # "hardworking" -> two pieces
    assert len(cont) == 2
    got = adapter.continuation_logprob(prompt, cont, cache=cache)
    logp = torch.log_softmax(_full_logits(adapter, ctx + prompt + cont), dim=-1)
    start = len(ctx) + len(prompt) - 1
    want = float(logp[start, cont[0]] + logp[start + 1, cont[1]])
    assert abs(got - want) < 1e-4


def test_graft_matches_native_layer0_keys(adapter):
    peer = adapter.new_cache("peer")
    adapter.forward(adapter.encode("q0 answer a0 . the g2a person is neg2 ."), peer)
    span = list(range(4, peer.get_seq_length()))
    tokens = [peer.provenance[t].token_id for t in span]

    recv = adapter.new_cache("recv")
    prefix = adapter.encode("they said one and one met there , they said .")
    adapter.forward(prefix, recv)
    assert recv.graft(peer, span, adapter) == len(span)
    native = adapter.new_cache("native")
    adapter.forward(prefix + tokens, native)
    got = recv.layer_keys(0)[:, len(prefix):, :]
    want = native.layer_keys(0)[:, len(prefix):, :]
    assert torch.allclose(got, want, atol=1e-4)
    assert recv.provenance[len(prefix)].channel == "peer"
    assert recv.provenance[len(prefix)].source_position == span[0]


def test_perturbation_is_read_time_only(adapter):
    cache = adapter.new_cache("p")
    adapter.forward(adapter.encode("the g0a person is neg0 ."), cache)
    before = cache.layer_values(1).clone()
    w = np.ones(adapter.head_dim)
    spec = InterventionSpec("perturb", {(1, 0): [2, 3]}, {(1, 0): w}, epsilon=1.0)
    plain = adapter.next_token_logits(adapter.encode("they said"), cache=cache)
    edited = adapter.next_token_logits(adapter.encode("they said"), cache=cache, intervention=spec)
    assert torch.equal(cache.layer_values(1), before)
    assert not np.allclose(plain, edited)


def test_conversation_runs_in_all_three_arms(adapter):
    cfg = adapter.cfg
    cfg.run.n_turns = 4
    cfg.run.max_new_tokens = 3
    orch = ConversationOrchestrator(adapter, cfg)
    lengths = {}
    for arm in (ARM_SHARED, ARM_TEXT, ARM_CONTROL):
        tr = orch.run(arm, 0.5, 0)
        lengths[arm] = tr.stats["seq_len"]
        assert len(tr.turns) == 4 and -1.0 <= tr.final_bias <= 1.0
        assert tr.values.shape[:2] == (adapter.n_layers, adapter.n_kv_heads)
        assert all(ch in ("system", "probe", "agent", "peer") for ch in
                   (p["channel"] for p in tr.provenance))
    assert lengths[ARM_SHARED] > lengths[ARM_TEXT] > lengths[ARM_CONTROL]
    probe = orch.probes.pick(0)
    assert isinstance(hallucinated(adapter, adapter.new_cache("h"), orch.probes.pick_fact(0)), bool)
    assert -1.0 <= bias_index(adapter, None, probe) <= 1.0


def test_generation_never_emits_special_tokens(adapter):
    out = adapter.generate(adapter.encode("they said"), adapter.new_cache("g"), max_new_tokens=8)
    assert len(out.token_ids) == 8 and not set(out.token_ids) & {0, 1}


def test_reference_codec_is_the_closed_vocabulary():
    from rcm.adapters import TinyAdapter

    ad = TinyAdapter(ExperimentConfig())
    text = "the g0a person is neg0 ."
    assert ad.encode(text) == encode(text) and ad.bos_ids() == [BOS]
    assert ad.word_ids("pos2") == [STOI["pos2"]] and ad.token_text(STOI["g1b"]) == "g1b"
    assert len(ProbeBank()) == 12


def test_raw_kv_sample_is_stratified_by_arm():
    cfg = ExperimentConfig()
    picked = [(a, p, s) for a in cfg.run.arms for p in cfg.run.pressures for s in cfg.run.seeds
              if raw_kv_selected(cfg, a, p, s)]
    assert len(picked) == 6                       # round(0.05 * 45) = 2 per arm
    for arm in cfg.run.arms:
        mine = [(p, s) for a, p, s in picked if a == arm]
        assert mine == [(0.0, 0), (1.0, 4)]
    assert not raw_kv_selected(cfg, ARM_SHARED, 0.3, 0)
