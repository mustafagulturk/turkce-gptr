"""Eğitim, checkpoint, ince ayar ve değerlendirme testleri (küçük sahte veriyle, CPU'da hızlı)."""

import csv
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from turkce_gpt.checkpoint import load_checkpoint, save_checkpoint
from turkce_gpt.config import make_config
from turkce_gpt.evaluate import bits_per_byte, evaluate_tokens, token_byte_lengths
from turkce_gpt.model import GPT
from turkce_gpt.tokenizer import END_OF_TEXT, BPETokenizer
from turkce_gpt.train import get_lr

TEXT = (
    "Bir varmış bir yokmuş, evvel zaman içinde kalbur saman içinde bir köy varmış. "
    "Köyde yaşlı bir kadın ile torunu yaşarmış. Her sabah erkenden kalkar, "
    "bahçedeki ağaçları sularlarmış. Bugün hava çok güzel, çocuklar parkta oynuyor. "
)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train([TEXT] * 5, vocab_size=320)
    return tok


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, tokenizer):
    """data.py'nin çıktı biçiminde küçük sahte bir veri klasörü."""
    d = tmp_path_factory.mktemp("hazir")
    end = tokenizer.special_id(END_OF_TEXT)
    ids = (tokenizer.encode(TEXT) + [end]) * 40
    arr = np.array(ids, dtype=np.uint16)
    split = int(len(arr) * 0.9)
    arr[:split].tofile(d / "train.bin")
    arr[split:].tofile(d / "val.bin")
    tokenizer.save(d / "tokenizer.json")
    meta = {"vocab_size": tokenizer.vocab_size, "kaynak": "test",
            "train_tokens": split, "val_tokens": len(arr) - split}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def tiny_model(vocab_size: int) -> GPT:
    torch.manual_seed(0)
    return GPT(make_config("deneme", vocab_size, block_size=32))


# ------------------------------------------------------------- lr takvimi
def test_lr_warmup_is_linear():
    lrs = [get_lr(s, 1e-3, 1e-4, warmup_steps=10, max_steps=100) for s in range(10)]
    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[9] == pytest.approx(1e-3)
    assert all(b > a for a, b in zip(lrs, lrs[1:]))


def test_lr_cosine_decay():
    max_lr, min_lr = 1e-3, 1e-4
    assert get_lr(10, max_lr, min_lr, 10, 110) == pytest.approx(max_lr)
    # Yolun yarısında tam ortada olmalı (cos(pi/2) = 0).
    assert get_lr(60, max_lr, min_lr, 10, 110) == pytest.approx((max_lr + min_lr) / 2)
    decay = [get_lr(s, max_lr, min_lr, 10, 110) for s in range(10, 110)]
    assert all(b <= a for a, b in zip(decay, decay[1:]))
    assert get_lr(110, max_lr, min_lr, 10, 110) == pytest.approx(min_lr)
    assert get_lr(500, max_lr, min_lr, 10, 110) == pytest.approx(min_lr)


# ------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_same_logits(tmp_path, tokenizer):
    model = tiny_model(tokenizer.vocab_size)
    model.eval()
    idx = torch.randint(0, tokenizer.vocab_size, (2, 16))
    with torch.no_grad():
        expected, _ = model(idx)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tokenizer, {"step": 7, "val_loss": 1.5})
    loaded, tok2, extra = load_checkpoint(path)

    assert not loaded.training
    assert extra["step"] == 7 and extra["val_loss"] == 1.5
    assert loaded.config == model.config
    with torch.no_grad():
        got, _ = loaded(idx)
    assert torch.allclose(expected, got)
    # Tokenizer da birebir aynı olmalı.
    assert tok2.encode("Bugün hava çok güzel.") == tokenizer.encode("Bugün hava çok güzel.")
    assert tok2.special_tokens == tokenizer.special_tokens


# ------------------------------------------------------------- egit cli
def test_train_cli_writes_model_and_log(tmp_path, data_dir, capsys):
    from turkce_gpt.train import cli

    out = tmp_path / "model"
    cli(["--veri", str(data_dir), "--boyut", "deneme", "--blok", "32", "--cikti", str(out),
         "--adim", "6", "--batch", "4", "--birikim", "2", "--isinma", "2",
         "--degerlendirme-araligi", "3", "--degerlendirme-batch", "2", "--log-araligi", "2",
         "--ornek-uzunluk", "5", "--cihaz", "cpu"])

    assert (out / "model.pt").exists()
    with open(out / "log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["adim"]) for r in rows] == [3, 6]
    assert set(rows[0]) == {"adim", "train_loss", "val_loss", "lr", "sure"}
    assert all(math.isfinite(float(r["val_loss"])) for r in rows)

    printed = capsys.readouterr().out
    assert "token/sn" in printed and "kalan" in printed

    model, _, extra = load_checkpoint(out / "model.pt")
    assert model.config.block_size == 32
    assert extra["val_loss"] == pytest.approx(min(float(r["val_loss"]) for r in rows), abs=1e-4)

    # --devam: kaldığı adımdan (6) devam edip 8'e kadar eğitir.
    cli(["--veri", str(data_dir), "--boyut", "deneme", "--cikti", str(out), "--adim", "8",
         "--batch", "4", "--isinma", "2", "--degerlendirme-araligi", "100", "--degerlendirme-batch", "2",
         "--ornek-uzunluk", "0", "--cihaz", "cpu", "--devam"])
    with open(out / "log.csv", encoding="utf-8") as f:
        assert [int(r["adim"]) for r in csv.DictReader(f)] == [3, 6, 8]


def test_train_cli_starts_from_saved_model(tmp_path, data_dir):
    from turkce_gpt.train import cli

    base = tmp_path / "temel"
    common = ["--veri", str(data_dir), "--boyut", "deneme", "--blok", "32", "--batch", "4", "--isinma", "1",
              "--degerlendirme-araligi", "4", "--degerlendirme-batch", "2", "--ornek-uzunluk", "0",
              "--cihaz", "cpu"]
    cli(common + ["--cikti", str(base), "--adim", "4"])
    first, _, _ = load_checkpoint(base / "model.pt")

    # Kayıtlı modelin ağırlıklarından başlayıp aynı mimariyle eğitime devam eder.
    cont = tmp_path / "devam"
    cli(common + ["--cikti", str(cont), "--adim", "4", "--baslangic-modeli", str(base / "model.pt")])
    second, _, _ = load_checkpoint(cont / "model.pt")
    assert second.config == first.config
    with open(cont / "log.csv", encoding="utf-8") as f:
        assert [int(r["adim"]) for r in csv.DictReader(f)] == [4]


def test_train_cli_rejects_different_tokenizer(tmp_path, data_dir):
    from turkce_gpt.train import cli

    other = BPETokenizer()
    other.train(["tamamen farklı bir metin, başka kelimeler"] * 3, vocab_size=300)
    model = tiny_model(other.vocab_size)
    save_checkpoint(tmp_path / "baska.pt", model, other, {})
    with pytest.raises(SystemExit):
        cli(["--veri", str(data_dir), "--cikti", str(tmp_path / "x"), "--adim", "1", "--cihaz", "cpu",
             "--baslangic-modeli", str(tmp_path / "baska.pt")])


# ------------------------------------------------------------- ince-ayar cli
def test_finetune_cli_runs_on_small_jsonl(tmp_path, tokenizer):
    base = tmp_path / "base.pt"
    save_checkpoint(base, tiny_model(tokenizer.vocab_size), tokenizer, {"step": 0, "val_loss": 5.0})

    conversations = [
        {"mesajlar": [{"role": "kullanici", "content": "Merhaba"}, {"role": "asistan", "content": "Merhaba, nasılsın?"}]},
        {"mesajlar": [{"role": "kullanici", "content": "Hava nasıl?"}, {"role": "asistan", "content": "Bugün hava çok güzel."}]},
        {"mesajlar": [{"role": "kullanici", "content": "Masal anlat"},
                      {"role": "asistan", "content": "Bir varmış bir yokmuş."},
                      {"role": "kullanici", "content": "Devam et"},
                      {"role": "asistan", "content": "Köyde yaşlı bir kadın yaşarmış."}]},
    ]
    jsonl = tmp_path / "sohbet.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in conversations) + "\n", encoding="utf-8")

    from turkce_gpt.finetune import cli

    out = tmp_path / "sohbet"
    cli(["--model", str(base), "--veri", str(jsonl), "--cikti", str(out), "--adim", "8",
         "--batch", "2", "--lr", "1e-3", "--isinma", "2", "--degerlendirme-araligi", "4", "--cihaz", "cpu"])

    model, _, extra = load_checkpoint(out / "model.pt")
    assert extra["tur"] == "sohbet"
    assert extra["val_loss"] < 5.8  # rastgele modelin ln(320) ≈ 5.77 civarından düşmeli
    assert (out / "log.csv").exists()


def test_finetune_collate_pads_and_shifts(tokenizer):
    from turkce_gpt.finetune import collate

    examples = [([1, 2, 3, 4], [-100, -100, 3, 4]), ([5, 6], [-100, 6])]
    x, y = collate(examples, pad_id=0, device="cpu")
    assert x.tolist() == [[1, 2, 3], [5, 0, 0]]
    assert y.tolist() == [[-100, 3, 4], [6, -100, -100]]


# ------------------------------------------------------------- degerlendir
def test_bits_per_byte_formula():
    # 4 byte için toplam 8 bit (= 8 * ln2 nat) -> byte başına 2 bit.
    assert bits_per_byte(8 * math.log(2), 4) == pytest.approx(2.0)


def test_evaluate_tokens_matches_manual_computation(tokenizer):
    model = tiny_model(tokenizer.vocab_size)
    end = tokenizer.special_id(END_OF_TEXT)
    tokens = np.array((tokenizer.encode(TEXT) + [end]) * 3, dtype=np.uint16)
    block = model.config.block_size
    result = evaluate_tokens(model, tokens, tokenizer, batch_size=2)

    n_windows = (len(tokens) - 1) // block
    x = torch.tensor(np.stack([tokens[i * block:(i + 1) * block] for i in range(n_windows)]).astype(np.int64))
    y = torch.tensor(np.stack([tokens[i * block + 1:(i + 1) * block + 1] for i in range(n_windows)]).astype(np.int64))
    with torch.no_grad():
        _, loss = model(x, y)
    assert result["loss"] == pytest.approx(loss.item(), rel=1e-4)
    assert result["perplexity"] == pytest.approx(math.exp(loss.item()), rel=1e-4)

    # Byte sayısı: her hedef tokenin UTF-8 uzunluğu, özel tokenler 1 byte.
    lengths = token_byte_lengths(tokenizer)
    assert lengths[end] == 1
    expected_bytes = int(lengths[y.numpy()].sum())
    assert result["bytes"] == expected_bytes
    expected_bpb = loss.item() * y.numel() / math.log(2) / expected_bytes
    assert result["bits_per_byte"] == pytest.approx(expected_bpb, rel=1e-4)


def test_evaluate_cli(tmp_path, data_dir, tokenizer, capsys):
    path = tmp_path / "model.pt"
    save_checkpoint(path, tiny_model(tokenizer.vocab_size), tokenizer, {"step": 1})
    from turkce_gpt.evaluate import cli

    cli(["--model", str(path), "--veri", str(data_dir), "--cihaz", "cpu"])
    printed = capsys.readouterr().out
    assert "Perplexity" in printed and "Bits-per-byte" in printed
