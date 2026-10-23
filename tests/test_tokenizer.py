"""BPE tokenizer testleri: kodla -> çöz aynı metni vermeli, özel tokenler doğru davranmalı."""

from __future__ import annotations

import pytest

from turkce_gpt.tokenizer import ASSISTANT, END_OF_TEXT, USER, BPETokenizer

# Küçük ama Türkçe ekleri tekrar eden bir eğitim metni.
TRAIN_TEXTS = [
    "Evlerimizden çıktık, okullarımıza gittik. Kitaplarımızı okuduk.",
    "Çocuklar bahçede oynuyorlar; öğretmenler sınıfta bekliyorlar.",
    "İstanbul'da yaşıyorum ama Gaziantep'i çok seviyorum.",
    "Işık, ılık, ıspanak, iğne, İzmir, ırmak ve ağaç.",
    "Güneşli günlerde şehirde gezmek güzeldir. Şemsiye almayı unutmayın!",
] * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train(TRAIN_TEXTS, vocab_size=256 + 3 + 120)
    return t


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Merhaba dünya",
        "çÇğĞıIİiöÖşŞüÜ",
        "İSTANBUL ırmak IŞIK iğne",
        "Bugün hava 🌞 çok güzel 😀👍🏽",
        "2026 yılında 3,14 ve 1.000.000 lira; 007!",
        "  çok   boşluklu\n\nsatırlar\t sekme  ",
        "Görülmemiş kelime: Muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine",
    ],
)
def test_roundtrip(tok: BPETokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_empty_text(tok: BPETokenizer) -> None:
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_vocab_size_and_special_ids(tok: BPETokenizer) -> None:
    assert tok.vocab_size == 256 + len(tok.merges) + 3
    ids = {tok.special_id(t) for t in (END_OF_TEXT, USER, ASSISTANT)}
    assert len(ids) == 3
    assert all(256 + len(tok.merges) <= i < tok.vocab_size for i in ids)


def test_special_tokens_allowed(tok: BPETokenizer) -> None:
    text = f"{USER}Selam{ASSISTANT}Merhaba!{END_OF_TEXT}"
    ids = tok.encode(text, allow_special=True)
    assert ids[0] == tok.special_id(USER)
    assert ids[-1] == tok.special_id(END_OF_TEXT)
    assert tok.special_id(ASSISTANT) in ids
    assert tok.decode(ids) == text


def test_special_tokens_not_allowed(tok: BPETokenizer) -> None:
    # Kullanıcı "<|son|>" yazarsa sıradan metin gibi kodlanmalı (güvenlik).
    text = f"bak {END_OF_TEXT} yazdım"
    ids = tok.encode(text)  # allow_special=False varsayılan
    specials = set(tok.special_tokens.values())
    assert not specials & set(ids)
    assert tok.decode(ids) == text


def test_save_load_same_encoding(tok: BPETokenizer, tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.special_tokens == tok.special_tokens
    for text in TRAIN_TEXTS[:5] + [f"Selam{END_OF_TEXT}", "yeni 🌍 metin"]:
        assert loaded.encode(text, allow_special=True) == tok.encode(text, allow_special=True)


def test_merges_learn_frequent_pieces() -> None:
    # " kitaplar" çok sık, "zqx" gibi parçalar seyrek geçiyor. BPE sık geçen
    # kelimeyi tek tokene indirmeli, seyrek olanı byte'larına bırakmalı.
    texts = ["Burada kitaplar var ve masada kitaplar duruyor."] * 50 + ["zqx wvy"]
    t = BPETokenizer()
    t.train(texts, vocab_size=256 + 3 + 40)
    assert len(t.encode(" kitaplar")) == 1
    assert " kitaplar".encode() in {t.vocab[i] for i in t.merges.values()}
    assert len(t.encode(" zqx")) == 4  # boşluk + 3 harf, hiç birleştirilmemiş
    # Sıkıştırma: eğitim metni byte sayısından belirgin şekilde az tokene kodlanmalı.
    text = texts[0]
    assert len(t.encode(text)) < 0.5 * len(text.encode("utf-8"))


def test_training_is_deterministic() -> None:
    a, b = BPETokenizer(), BPETokenizer()
    a.train(TRAIN_TEXTS, vocab_size=300)
    b.train(TRAIN_TEXTS, vocab_size=300)
    assert a.merges == b.merges


def test_too_small_vocab_raises() -> None:
    with pytest.raises(ValueError):
        BPETokenizer().train(["abc"], vocab_size=100)
