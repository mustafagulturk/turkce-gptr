"""interactive.py ve veri/sohbet.jsonl testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turkce_gpt.interactive import (
    IncrementalDecoder,
    build_chat_ids,
    build_chat_parser,
    build_generate_parser,
    decode_stream,
    strip_special_tokens,
)
from turkce_gpt.tokenizer import SPECIAL_TOKENS, BPETokenizer

SOHBET_JSONL = Path(__file__).resolve().parent.parent / "veri" / "sohbet.jsonl"
TURKISH_TEXT = "Şişli'de güzel bir öğleden sonra; çiçekler açmış, İğdır'dan ılık rüzgâr esiyor. €"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train([TURKISH_TEXT] * 3 + ["Merhaba dünya, nasılsın?"], vocab_size=300)
    return tok


# ------------------------------------------------------------ artımlı çözme
def test_byte_tokens_never_emit_half_characters(tokenizer: BPETokenizer) -> None:
    # Hiç birleştirme uygulamadan ham byte'ları tek tek veriyoruz: her çok
    # byte'lı karakter (ş, ğ, İ, ı, €) kesin olarak birden çok tokene bölünür.
    raw = list(TURKISH_TEXT.encode("utf-8"))
    pieces = decode_stream(tokenizer, raw)
    assert "".join(pieces) == TURKISH_TEXT
    assert all("�" not in p for p in pieces)  # hiçbir adımda '�' basılmadı


def test_two_byte_character_waits_for_second_byte(tokenizer: BPETokenizer) -> None:
    first, second = "ş".encode("utf-8")  # 0xC5 0x9F
    decoder = IncrementalDecoder(tokenizer)
    assert decoder.add(first) == ""     # yarım: bekle
    assert decoder.add(second) == "ş"   # tamamlandı
    assert decoder.flush() == ""


def test_three_byte_character(tokenizer: BPETokenizer) -> None:
    b1, b2, b3 = "€".encode("utf-8")
    decoder = IncrementalDecoder(tokenizer)
    assert [decoder.add(b1), decoder.add(b2), decoder.add(b3)] == ["", "", "€"]


def test_stream_matches_full_decode_with_bpe_tokens(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode(TURKISH_TEXT)
    pieces = decode_stream(tokenizer, ids)
    assert "".join(pieces) == tokenizer.decode(ids) == TURKISH_TEXT


def test_incomplete_tail_is_replaced_on_flush(tokenizer: BPETokenizer) -> None:
    # Model yarım bir karakterde durursa flush '�' döndürür; program çökmez.
    decoder = IncrementalDecoder(tokenizer)
    assert decoder.add("a".encode()[0]) == "a"
    assert decoder.add("ğ".encode("utf-8")[0]) == ""
    assert decoder.flush() == "�"


def test_strip_special_tokens() -> None:
    assert strip_special_tokens("Selam<|son|> <|asistan|>dünya<|kullanici|>") == "Selam dünya"


def test_build_chat_ids_drops_oldest_turns(tokenizer: BPETokenizer) -> None:
    history = []
    for i in range(10):
        history.append({"role": "kullanici", "content": f"Soru {i}: nasılsın?"})
        history.append({"role": "asistan", "content": "Merhaba dünya, iyiyim."})
    history.append({"role": "kullanici", "content": "Son soru"})
    ids = build_chat_ids(tokenizer, history, max_prompt_tokens=40)
    assert len(ids) <= 40
    assert history[-1]["content"] == "Son soru"          # son mesaj korunur
    assert history[0]["role"] == "kullanici"              # turlar çift çift atılır
    assert len(history) < 21


# ------------------------------------------------------------ argparse
def test_generate_defaults() -> None:
    args = build_generate_parser().parse_args(["--model", "m.pt"])
    assert args.baslangic == "Bir varmış"
    assert args.uzunluk == 200
    assert args.sicaklik == 0.8
    assert args.top_k == 50
    assert args.top_p == 0.95
    assert args.tohum == 42
    assert args.cihaz == "auto"


def test_chat_defaults() -> None:
    args = build_chat_parser().parse_args(["--model", "m.pt"])
    assert args.model == "m.pt"
    assert args.uzunluk == 200
    assert args.top_k == 50
    assert args.cihaz == "auto"


def test_model_is_required() -> None:
    with pytest.raises(SystemExit):
        build_generate_parser().parse_args([])


# ------------------------------------------------------------ sohbet.jsonl
def _load_jsonl() -> list[dict]:
    lines = SOHBET_JSONL.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_jsonl_has_enough_examples() -> None:
    assert len(_load_jsonl()) >= 140


def test_jsonl_every_line_follows_template() -> None:
    for n, row in enumerate(_load_jsonl(), start=1):
        assert set(row) == {"mesajlar"}, f"{n}. satır"
        messages = row["mesajlar"]
        # Kullanıcı ile başlar, asistanla biter, roller sırayla değişir.
        assert len(messages) >= 2 and len(messages) % 2 == 0, f"{n}. satır"
        for i, msg in enumerate(messages):
            assert set(msg) == {"role", "content"}, f"{n}. satır"
            assert msg["role"] == ("kullanici" if i % 2 == 0 else "asistan"), f"{n}. satır"
            assert isinstance(msg["content"], str) and msg["content"].strip(), f"{n}. satır"
            assert msg["content"] == msg["content"].strip(), f"{n}. satır"
            assert not any(t in msg["content"] for t in SPECIAL_TOKENS), f"{n}. satır"


def test_jsonl_works_with_chat_template(tokenizer: BPETokenizer) -> None:
    from turkce_gpt.chat import encode_conversation

    for row in _load_jsonl():
        ids, labels = encode_conversation(tokenizer, row["mesajlar"])
        assert len(ids) == len(labels)
        assert any(label != -100 for label in labels)  # öğrenilecek bir cevap var
