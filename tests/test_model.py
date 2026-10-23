"""model.py testleri: şekiller, nedensellik, KV cache, üretim ve optimizer."""

from __future__ import annotations

import math

import pytest
import torch

from turkce_gpt.config import GPTConfig
from turkce_gpt.model import GPT, KVCache

VOCAB = 97


def make_model(dropout: float = 0.0, block_size: int = 32, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=VOCAB, block_size=block_size, n_layer=2, n_head=2,
                       n_embd=32, dropout=dropout)
    return GPT(config).eval()


def set_sdpa(model: GPT, flag: bool) -> None:
    for block in model.blocks:
        block.attn.use_sdpa = flag


def test_output_shapes():
    model = make_model()
    idx = torch.randint(0, VOCAB, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, VOCAB)
    assert loss is None
    logits, loss = model(idx, idx)
    assert loss is not None and loss.dim() == 0
    assert model.num_params() > 0


def test_weight_tying():
    model = make_model()
    assert model.lm_head.weight is model.tok_emb.weight


def test_initial_loss_near_ln_vocab():
    model = make_model()
    idx = torch.randint(0, VOCAB, (8, 32))
    _, loss = model(idx, torch.randint(0, VOCAB, (8, 32)))
    # Başlangıçta model her tokene ~eşit olasılık vermeli.
    assert abs(loss.item() - math.log(VOCAB)) < 0.3


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_causality(use_sdpa: bool):
    model = make_model()
    set_sdpa(model, use_sdpa)
    idx = torch.randint(0, VOCAB, (1, 16))
    logits_a, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 10] = (idx2[0, 10] + 1) % VOCAB  # 10. konumdaki tokeni değiştir
    logits_b, _ = model(idx2)
    # 10'dan önceki konumlar etkilenmemeli, 10 ve sonrası etkilenmeli.
    assert torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-6)
    assert not torch.allclose(logits_a[0, 10:], logits_b[0, 10:])


def test_manual_attention_matches_sdpa():
    model = make_model()
    idx = torch.randint(0, VOCAB, (2, 20))
    set_sdpa(model, True)
    a, _ = model(idx)
    set_sdpa(model, False)
    b, _ = model(idx)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_kv_cache_matches_full_forward(use_sdpa: bool):
    model = make_model()
    set_sdpa(model, use_sdpa)
    idx = torch.randint(0, VOCAB, (1, 12))
    full, _ = model(idx)

    # Adım adım: önce 5 tokenlik ön ek, sonra 3'lü bir parça, sonra tek tek.
    cache = KVCache(model.config.n_layer)
    parts = [model(idx[:, :5], kv_cache=cache)[0]]
    assert len(cache) == 5
    parts.append(model(idx[:, 5:8], kv_cache=cache)[0])
    for t in range(8, 12):
        parts.append(model(idx[:, t:t + 1], kv_cache=cache)[0])
    assert len(cache) == 12
    stepwise = torch.cat(parts, dim=1)
    assert torch.allclose(full, stepwise, atol=1e-5)


def test_ignore_index():
    model = make_model()
    idx = torch.randint(0, VOCAB, (2, 8))
    targets = torch.randint(0, VOCAB, (2, 8))
    logits, loss = model(idx, targets)
    masked = targets.clone()
    masked[:, :4] = -100
    _, loss_masked = model(idx, masked)
    expected = torch.nn.functional.cross_entropy(logits[:, 4:].reshape(-1, VOCAB), targets[:, 4:].reshape(-1))
    assert torch.allclose(loss_masked, expected, atol=1e-5)
    assert not torch.allclose(loss, loss_masked)


def test_overfit_small_batch():
    model = make_model(dropout=0.1).train()
    torch.manual_seed(1)
    data = torch.randint(0, VOCAB, (4, 17))
    x, y = data[:, :-1], data[:, 1:]
    opt = model.configure_optimizer(weight_decay=0.0, lr=1e-2)
    _, first = model(x, y)
    for _ in range(60):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.3


def test_generate_greedy_deterministic_and_stop():
    model = make_model()
    idx = torch.randint(0, VOCAB, (1, 5))
    a = list(model.generate_stream(idx, 10, temperature=0))
    b = list(model.generate_stream(idx, 10, temperature=0))
    assert a == b and len(a) == 10
    # Greedy üretim, tam forward ile argmax'a eşit olmalı (cache doğruluğu).
    seq = idx
    for tok in a[:4]:
        logits, _ = model(seq)
        assert int(logits[0, -1].argmax()) == tok
        seq = torch.cat([seq, torch.tensor([[tok]])], dim=1)
    # 3. üretilen token stop olarak verilirse ondan önce durmalı (onu yield etmeden).
    stop = a[2]
    first_pos = a.index(stop)
    c = list(model.generate_stream(idx, 10, temperature=0, stop_ids={stop}))
    assert c == a[:first_pos]
    out = model.generate(idx, 10, temperature=0)
    assert out.shape == (1, 15) and out[0, 5:].tolist() == a


def test_generate_sampling_options():
    model = make_model()
    idx = torch.randint(0, VOCAB, (1, 5))
    torch.manual_seed(0)
    out = list(model.generate_stream(idx, 8, temperature=1.0, top_k=5, top_p=0.9, repetition_penalty=1.3))
    assert len(out) == 8 and all(0 <= t < VOCAB for t in out)


def test_generate_beyond_block_size():
    model = make_model(block_size=16)
    # Başlangıç bile block_size'dan uzun olabilir; hata vermemeli.
    idx = torch.randint(0, VOCAB, (1, 20))
    out = list(model.generate_stream(idx, 30, temperature=0))
    assert len(out) == 30
    out = model.generate(torch.randint(0, VOCAB, (1, 3)), 40, temperature=0.8)
    assert out.shape == (1, 43)


def test_forward_too_long_raises():
    model = make_model(block_size=16)
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (1, 17)))


def test_generate_restores_training_mode():
    model = make_model().train()
    list(model.generate_stream(torch.zeros(1, 3, dtype=torch.long), 3))
    assert model.training


def test_configure_optimizer_groups():
    model = make_model()
    opt = model.configure_optimizer(weight_decay=0.1, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    # Her parametre tam olarak bir grupta (paylaşılan embedding tek sefer).
    n_grouped = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_grouped == model.num_params()
    assert opt.defaults["betas"] == (0.9, 0.95)
