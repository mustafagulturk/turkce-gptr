"""Veri hazırlama testleri: temizlik, küçük uçtan uca prepare ve TokenDataset. Ağ kullanılmaz."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from turkce_gpt.data import (
    TokenDataset,
    clean_document,
    clean_line,
    deduplicate,
    file_documents,
    join_sentences,
    prepare,
    read_conllu_sentences,
    split_documents,
)
from turkce_gpt.tokenizer import END_OF_TEXT, BPETokenizer


# ------------------------------------------------------------------ temizlik
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Annesi ile ( annesiyle ) babası geldiler .", "Annesi ile (annesiyle) babası geldiler."),
        ("Elma , armut ; kiraz : hepsi var !", "Elma, armut; kiraz: hepsi var!"),
        ("Neden ?  Çünkü   öyle .", "Neden? Çünkü öyle."),
        ('Bir hafta sonra yeniden " Pembe Konak " a geldim .', 'Bir hafta sonra yeniden "Pembe Konak" a geldim.'),
        ("%50 indirim ( sadece bugün )", "%50 indirim (sadece bugün)"),
        ("  baştaki ve sondaki boşluk  ", "baştaki ve sondaki boşluk"),
    ],
)
def test_clean_line(raw: str, expected: str) -> None:
    assert clean_line(raw) == expected


def test_clean_line_nfc() -> None:
    decomposed = "İstanbul şehri"  # "İ" ve "ş" iki parçalı (NFD) yazılmış
    assert clean_line(decomposed) == "İstanbul şehri"


def test_clean_line_keeps_apostrophe_and_odd_quotes() -> None:
    assert clean_line("Ankara'da kaldık .") == "Ankara'da kaldık."
    assert clean_line('Tek " tırnak') == 'Tek " tırnak'  # eşleşmeyen tırnağa dokunma


def test_clean_document_drops_short_lines() -> None:
    doc = "Bu yeterince uzun bir satır.\n.\n12345\n\nİkinci uzun satır da burada."
    assert clean_document(doc) == "Bu yeterince uzun bir satır.\nİkinci uzun satır da burada."
    assert clean_document(". .\n1.") == ""


def test_deduplicate() -> None:
    assert list(deduplicate(["a b c", "d e f", "a b c"])) == ["a b c", "d e f"]


def test_split_documents_is_seeded_and_disjoint() -> None:
    docs = [f"belge {i}" for i in range(100)]
    tr1, va1 = split_documents(docs, 0.1, seed=1)
    tr2, va2 = split_documents(docs, 0.1, seed=1)
    assert (tr1, va1) == (tr2, va2)
    assert len(va1) == 10 and len(tr1) == 90
    assert set(tr1).isdisjoint(va1)
    # Çok küçük oranda bile doğrulamada en az bir belge olmalı.
    assert len(split_documents(docs[:5], 0.001, seed=0)[1]) == 1


def test_read_conllu_sentences(tmp_path) -> None:
    conllu = tmp_path / "x.conllu"
    conllu.write_text(
        "# newdoc id = 1\n# sent_id = 1\n# text = Birinci cümle .\n1\tBirinci\n\n"
        "# sent_id = 2\n# text = İkinci cümle .\n1\tİkinci\n\n"
        "# newdoc id = 2\n# sent_id = 3\n# text = Başka belge .\n1\tBaşka\n\n",
        encoding="utf-8",
    )
    assert read_conllu_sentences(conllu) == [["Birinci cümle .", "İkinci cümle ."], ["Başka belge ."]]


def test_join_sentences_keeps_boundaries() -> None:
    # Noktasız biten cümleden sonra satır sonu konur ki cümle sınırı kaybolmasın.
    assert join_sentences(["Merhaba dünya.", "uçuşları göster", "Tamam."]) == "Merhaba dünya. uçuşları göster\nTamam."


def test_file_documents_split_on_blank_lines(tmp_path) -> None:
    f = tmp_path / "metin.txt"
    f.write_text("birinci satır\naynı belge\n\n\nikinci belge\n", encoding="utf-8")
    assert list(file_documents(f)) == ["birinci satır\naynı belge", "ikinci belge"]


# ------------------------------------------------------------- uçtan uca
SENTENCES = [
    "Türkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur.",
    "Gaziantep baklavası ve kebabıyla ünlü bir şehirdir.",
    "Çocuklar okuldan sonra parkta top oynadılar.",
    "Öğretmenimiz bize ilginç bir hikâye anlattı.",
    "Kışın dağlara kar yağar, yazın ise deniz kenarı çok sıcak olur.",
    "Kitap okumak insanın kelime dağarcığını geliştirir.",
]


# 40 farklı belge (tekrar ayıklamaya takılmasınlar diye numaralı).
BLOCKS = [f"{i}. paragraf : {SENTENCES[i % 6]} {SENTENCES[(i + 1) % 6]}" for i in range(40)]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("veri")
    txt = tmp / "metin.txt"
    # Sona bir tekrar ve bir çok kısa belge ekliyoruz; ikisi de atılmalı.
    txt.write_text("\n\n".join(BLOCKS + [BLOCKS[0], "."]) + "\n", encoding="utf-8")
    out = tmp / "hazir"
    meta = prepare("dosya", out, vocab_size=400, val_ratio=0.1, seed=0, file_path=txt, workers=1)
    return out, meta


def test_prepare_outputs(prepared) -> None:
    out, meta = prepared
    for name in ("tokenizer.json", "train.bin", "val.bin", "meta.json"):
        assert (out / name).exists()
    on_disk = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    assert meta["kaynak"] == "dosya"
    assert meta["train_docs"] + meta["val_docs"] == 40  # tekrar ve kısa belge atıldı
    assert meta["val_docs"] == 4
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    val = np.fromfile(out / "val.bin", dtype=np.uint16)
    assert len(train) == meta["train_tokens"] and len(val) == meta["val_tokens"]
    assert train.max() < meta["vocab_size"]


def test_prepare_documents_separated_by_eot(prepared) -> None:
    out, meta = prepared
    tok = BPETokenizer.load(out / "tokenizer.json")
    assert tok.vocab_size == meta["vocab_size"]
    eot = tok.special_id(END_OF_TEXT)
    assert meta["eot_id"] == eot
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert train[-1] == eot
    assert int((train == eot).sum()) == meta["train_docs"]
    # İlk belge çözülünce dosyadaki (temizlenmiş) belgelerden biri çıkmalı.
    first_end = int(np.argmax(train == eot))
    first_doc = tok.decode(train[:first_end].tolist())
    assert first_doc in {clean_document(b) for b in BLOCKS}


def test_token_dataset_batches(prepared) -> None:
    out, meta = prepared
    ds = TokenDataset(out)
    assert ds.vocab_size == meta["vocab_size"]
    assert ds.num_tokens("train") == meta["train_tokens"]
    torch.manual_seed(0)
    x, y = ds.get_batch("train", batch_size=4, block_size=16, device="cpu")
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.int64
    # y, x'in bir kaydırılmış hali: x[:, 1:] == y[:, :-1]
    assert torch.equal(x[:, 1:], y[:, :-1])
    # Ve gerçekten dosyadaki ardışık tokenler:
    train = np.fromfile(out / "train.bin", dtype=np.uint16).astype(np.int64)
    row = x[0].tolist()
    starts = [i for i in range(len(train) - 16) if train[i : i + 16].tolist() == row]
    assert starts and train[starts[0] + 16] == y[0, -1].item()

    xv, yv = ds.get_batch("val", batch_size=2, block_size=8)
    assert xv.shape == (2, 8)


def test_token_dataset_errors(prepared, tmp_path) -> None:
    out, meta = prepared
    ds = TokenDataset(out)
    with pytest.raises(ValueError):
        ds.get_batch("val", batch_size=1, block_size=meta["val_tokens"])
    with pytest.raises(ValueError):
        ds.get_batch("test", batch_size=1, block_size=4)
    with pytest.raises(FileNotFoundError):
        TokenDataset(tmp_path / "yok")


def test_prepare_reuses_existing_tokenizer(prepared, tmp_path) -> None:
    # Kayıtlı bir modelin üstüne eğitmek için yeni veri ESKİ tokenizer'la kodlanmalı.
    from turkce_gpt.tokenizer import BPETokenizer

    out, _ = prepared
    txt = tmp_path / "yeni.txt"
    txt.write_text("\n\n".join(f"{i}. yeni belge : {SENTENCES[i % 6]}" for i in range(20)) + "\n",
                   encoding="utf-8")
    new_out = tmp_path / "yeni_hazir"
    meta = prepare("dosya", new_out, vocab_size=999, val_ratio=0.1, seed=0, file_path=txt, workers=1,
                   tokenizer_path=out / "tokenizer.json")
    old = BPETokenizer.load(out / "tokenizer.json")
    new = BPETokenizer.load(new_out / "tokenizer.json")
    assert new.merges == old.merges
    assert meta["vocab_size"] == old.vocab_size  # --sozluk yok sayıldı


def test_collect_documents_skip_mb(tmp_path) -> None:
    from turkce_gpt.data import collect_documents

    txt = tmp_path / "metin.txt"
    txt.write_text("\n\n".join(BLOCKS) + "\n", encoding="utf-8")
    all_docs = collect_documents("dosya", None, file_path=txt)
    first_mb = len(all_docs[0].encode("utf-8")) / (1024 * 1024)
    rest = collect_documents("dosya", None, file_path=txt, skip_mb=first_mb)
    assert rest == all_docs[1:]
