"""
Veri hazırlama: ham metin -> temiz belgeler -> tokenizer -> token dosyaları.

Bir dil modeli eğitmeden önce metni modelin "yiyebileceği" hale getirmemiz
gerekir. Bu dosya baştan sona şu boru hattını (pipeline) uygular:

    1. Kaynaktan metni al      (UD treebank'leri, Vikipedi ya da yerel .txt)
    2. Temizle                 (Unicode düzeltme, gereksiz boşluklar, tekrarlar)
    3. Belgeleri karıştır ve ayır (eğitim / doğrulama)
    4. Tokenizer eğit          (sadece eğitim belgeleriyle, BPE)
    5. Kodla                   (her belge -> token numaraları, araya <|son|>)
    6. Diske yaz               (train.bin, val.bin: uint16 düz dizi + meta.json)

Eğitim sırasında ise `TokenDataset` bu .bin dosyalarını np.memmap ile açar ve
rastgele yerlerden (x, y) parçaları keser. memmap sayesinde dosya RAM'e
tamamen yüklenmez; işletim sistemi sadece okunan kısmı belleğe getirir.

Komut satırından kullanım:
    python -m turkce_gpt veri --kaynak ud --cikti veri/hazir --sozluk 8192
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np

from .tokenizer import END_OF_TEXT, BPETokenizer

if TYPE_CHECKING:  # sadece tip ipuçları için; torch'u gereksiz yere yüklemeyelim
    import torch

# --------------------------------------------------------------------------
# Kaynak bilgileri ve lisanslar
# --------------------------------------------------------------------------
# Universal Dependencies (UD) Türkçe treebank'leri: dilbilimciler tarafından
# cümle cümle işaretlenmiş derlemler. Biz sadece cümlelerin düz metnini
# ("# text = ..." satırları) kullanıyoruz.
#
# LİSANSLAR (veriyi paylaşırken/yayınlarken uyulmalı):
#   - IMST     : CC BY-NC-SA 4.0  -> ticari olmayan kullanım şartı var!
#   - diğerleri: CC BY-SA 4.0     -> atıf ver, türetilmiş eseri aynı lisansla paylaş
# Bu bilgiler meta.json'a da yazılır ki eğitilen modelin hangi veriden
# geldiği ve hangi şartlara tabi olduğu kaybolmasın.
UD_TREEBANKS: dict[str, str] = {
    "BOUN": "CC BY-SA 4.0",
    "Kenet": "CC BY-SA 4.0",
    "Penn": "CC BY-SA 4.0",
    "IMST": "CC BY-NC-SA 4.0",
    "Tourism": "CC BY-SA 4.0",
    "FrameNet": "CC BY-SA 4.0",
    "Atis": "CC BY-SA 4.0",
    "GB": "CC BY-SA 4.0",
}
UD_URL = "https://github.com/UniversalDependencies/UD_Turkish-{name}.git"

# Vikipedi metinleri de CC BY-SA 4.0 lisanslıdır.
WIKIPEDIA_LICENSE = "CC BY-SA 4.0 (Vikipedi)"

# Temizlikte bu kadar karakterden kısa satırları/cümleleri atıyoruz.
# "." ya da "1." gibi satırlar modele bir şey öğretmez, sadece gürültüdür.
MIN_LINE_CHARS = 8


# --------------------------------------------------------------------------
# Temizleme
# --------------------------------------------------------------------------
# UD metinlerinde cümleler çoğu zaman token token yazılmıştır:
#     "Annesi ile ( annesiyle ) babası geldiler ."
# Modelin normal yazıyı öğrenmesi için bunları düzeltiyoruz:
#     "Annesi ile (annesiyle) babası geldiler."
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?%)\]}…])")  # " ." -> "."
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")            # "( x" -> "(x"
_MULTI_SPACE = re.compile(r"[ \t  -​　]+")  # her türlü yatay boşluk
_HAS_LETTER = re.compile(r"[^\W\d_]")                     # en az bir harf var mı?
_SENTENCE_END = set(".!?…:;\"')»")                         # cümle sonu sayılan karakterler


def _fix_double_quotes(line: str) -> str:
    """Çift tırnakların içindeki fazla boşlukları siler: '" merhaba "' -> '"merhaba"'.

    Tırnak açılış mı kapanış mı, tek başına bakınca anlaşılmaz. Bu yüzden
    sırayla sayıyoruz: 1., 3., 5. tırnak açılış, 2., 4. ... kapanıştır.
    Sayı tek ise (eksik tırnak) yanlış eşleştirme yapmamak için dokunmuyoruz.
    """
    if line.count('"') % 2 != 0:
        return line
    parts = line.split('"')
    # split sonrası: parts[1], parts[3], ... tırnak İÇİNDEKİ metinlerdir.
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].strip()
    return '"'.join(parts)


def clean_line(line: str) -> str:
    """Tek bir satırı normalleştirir ve boşluklarını düzeltir."""
    # NFC: "i" + "birleşik nokta" gibi iki parçalı yazılmış harfleri tek
    # karaktere çevirir. Aynı görünen iki metin farklı byte'lara sahip olursa
    # tokenizer onları farklı şeyler sanar; NFC bunu önler.
    line = unicodedata.normalize("NFC", line)
    line = _MULTI_SPACE.sub(" ", line).strip()
    line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
    line = _SPACE_AFTER_OPEN.sub(r"\1", line)
    line = _fix_double_quotes(line)
    return line


def is_useful_line(line: str, min_chars: int = MIN_LINE_CHARS) -> bool:
    """Satır modele bir şey öğretecek kadar uzun ve en az bir harf içeriyor mu?"""
    return len(line) >= min_chars and _HAS_LETTER.search(line) is not None


def clean_document(text: str, min_chars: int = MIN_LINE_CHARS) -> str:
    """Bir belgeyi satır satır temizler; işe yaramayan satırları atar.

    Satır sonlarını koruyoruz çünkü paragraf yapısı da modelin öğrenmesi
    gereken bir şeydir. Boş dönerse belge tamamen atılmalıdır.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (clean_line(ln) for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if is_useful_line(ln, min_chars))


def deduplicate(docs: Iterable[str]) -> Iterator[str]:
    """Aynı belgeyi ikinci kez görürsek atar.

    Tekrarlanan metin modeli o metni ezberlemeye iter ve doğrulama kaybını
    yanıltıcı biçimde düşürür (aynı metin hem train hem val'de olabilir).
    Belgenin kendisi yerine 16 byte'lık özetini (hash) saklayarak bellek
    tasarrufu yapıyoruz.
    """
    seen: set[bytes] = set()
    for doc in docs:
        key = hashlib.blake2b(doc.encode("utf-8"), digest_size=16).digest()
        if key not in seen:
            seen.add(key)
            yield doc


# --------------------------------------------------------------------------
# Kaynak 1: UD treebank'leri
# --------------------------------------------------------------------------
def download_ud(cache_dir: str | Path) -> list[Path]:
    """Treebank depolarını önbellek klasörüne indirir (zaten varsa atlar).

    `git clone --depth 1` sadece son sürümü indirir, geçmişi indirmez; hem
    hızlı hem de az yer kaplar. Dönen değer: tüm .conllu dosyaları.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for name in UD_TREEBANKS:
        repo_dir = cache_dir / f"UD_Turkish-{name}"
        if not repo_dir.exists():
            print(f"  indiriliyor: UD_Turkish-{name} ...", flush=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", UD_URL.format(name=name), str(repo_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"UD_Turkish-{name} indirilemedi (git clone hatası):\n{result.stderr.strip()}"
                )
        found = sorted(repo_dir.glob("*.conllu"))
        if not found:
            print(f"  uyarı: UD_Turkish-{name} içinde .conllu dosyası yok", flush=True)
        files.extend(found)
    return files


def read_conllu_sentences(path: str | Path) -> list[list[str]]:
    """Bir .conllu dosyasındaki cümleleri, belge sınırlarına göre gruplu döner.

    Dosyada "# newdoc" satırı varsa (ör. IMST) her belge ayrı bir listedir;
    yoksa bütün dosya tek bir liste olur. Cümle sırası korunur.
    """
    groups: list[list[str]] = [[]]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# newdoc") and groups[-1]:
                groups.append([])
            elif line.startswith("# text = "):
                groups[-1].append(line[len("# text = "):].strip())
    return [g for g in groups if g]


def ud_documents(cache_dir: str | Path, sentences_per_doc: int = 8) -> Iterator[str]:
    """UD cümlelerini "belge" gibi davranan küçük paragraflara çevirir.

    Neden gruplama? Tek tek cümlelerin her birinin arasına <|son|> koyarsak
    model sadece kısa cümleler görür, uzun bağlam kurmayı öğrenemez. Aynı
    dosyadaki ARDIŞIK cümleler (çoğu treebank'te aynı metinden gelir)
    birbiriyle ilişkili olduğundan onları bir paragrafta birleştiriyoruz.
    """
    files = download_ud(cache_dir)
    seen: set[str] = set()  # aynı cümle birden çok kez geçiyorsa (ATIS'te çok) bir kez al
    for path in files:
        for group in read_conllu_sentences(path):
            sentences = []
            for s in group:
                s = clean_line(s)
                if is_useful_line(s) and s not in seen:
                    seen.add(s)
                    sentences.append(s)
            for i in range(0, len(sentences), sentences_per_doc):
                yield join_sentences(sentences[i : i + sentences_per_doc])


def join_sentences(sentences: list[str]) -> str:
    """Cümleleri paragraf olarak birleştirir.

    Bazı treebank'lerde (ör. ATIS) cümleler noktasız biter. Bunları boşlukla
    yan yana koyarsak cümle sınırı kaybolur; bu yüzden noktalamayla bitmeyen
    bir cümleden sonra boşluk yerine satır sonu koyuyoruz.
    """
    out = sentences[0]
    for prev, sent in zip(sentences, sentences[1:]):
        out += (" " if prev[-1] in _SENTENCE_END else "\n") + sent
    return out


# --------------------------------------------------------------------------
# Kaynak 2: Vikipedi (Colab'da, HuggingFace üzerinden)
# --------------------------------------------------------------------------
def wikipedia_documents() -> Iterator[str]:
    """Türkçe Vikipedi makalelerini akış (streaming) halinde verir.

    streaming=True sayesinde ~1 GB'lık veri setinin tamamı indirilmez;
    makaleler okundukça gelir ve --max-mb sınırına gelince dururuz.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Hata: Vikipedi kaynağı için 'datasets' kütüphanesi gerekli.\n"
            "Kurmak için:  pip install datasets\n"
            "(İnternet erişimi olmayan bir makinedeyseniz --kaynak ud ya da --kaynak dosya kullanın.)"
        ) from None
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for row in ds:
        yield row["text"]


# --------------------------------------------------------------------------
# Kaynak 3: yerel metin dosyası
# --------------------------------------------------------------------------
def file_documents(path: str | Path) -> Iterator[str]:
    """Bir .txt dosyasını belgelere böler: boş satırla ayrılmış her blok bir belgedir."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Hata: dosya bulunamadı: {path}")
    block: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                block.append(line.rstrip("\n"))
            elif block:
                yield "\n".join(block)
                block = []
    if block:
        yield "\n".join(block)


# --------------------------------------------------------------------------
# Kodlama (paralel)
# --------------------------------------------------------------------------
# Çok çekirdekli kodlamada her işçi süreç tokenizer'ı bir kez yükler ve
# bu global değişkende tutar. (Her belgeyle birlikte tokenizer'ı göndermek
# çok pahalı olurdu.)
_worker_tokenizer: BPETokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer.load(tokenizer_path)


def _encode_with_worker(doc: str) -> np.ndarray:
    assert _worker_tokenizer is not None
    return encode_document(_worker_tokenizer, doc)


def encode_document(tokenizer: BPETokenizer, doc: str) -> np.ndarray:
    """Belgeyi kodlar ve sonuna <|son|> ekler.

    <|son|> modele "burada bir metin bitti, sonraki metin ilgisiz" der.
    Üretim sırasında da model bu tokeni üretince dururuz.
    """
    ids = tokenizer.encode(doc)  # belge metnindeki "<|son|>" yazıları özel token sayılmaz
    ids.append(tokenizer.special_id(END_OF_TEXT))
    return np.asarray(ids, dtype=np.uint16)


def encode_to_file(
    docs: list[str],
    tokenizer: BPETokenizer,
    tokenizer_path: Path,
    out_path: Path,
    workers: int,
    label: str,
) -> int:
    """Belgeleri kodlayıp uint16 olarak dosyaya yazar; toplam token sayısını döner.

    uint16 (2 byte) 0..65535 arası sayıları tutar; sözlüğümüz bundan küçük
    olduğu için int64'e (8 byte) göre dört kat daha az yer kaplar.
    Sonuçları bellekte biriktirmeden, geldikçe dosyaya ekliyoruz.
    """
    total_bytes = sum(len(d) for d in docs)
    # Küçük veride süreç başlatmanın maliyeti kazançtan büyük; tek çekirdek yeter.
    use_pool = workers > 1 and total_bytes > 5_000_000
    n_tokens = 0
    start = time.time()
    with open(out_path, "wb") as f:
        if use_pool:
            import multiprocessing as mp

            with mp.Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
                # imap sırayı korur: belgeler dosyaya karıştırılmış sırayla yazılır.
                results = pool.imap(_encode_with_worker, docs, chunksize=64)
                for i, arr in enumerate(results, 1):
                    arr.tofile(f)
                    n_tokens += len(arr)
                    if i % 20_000 == 0:
                        print(f"  {label}: {i:,}/{len(docs):,} belge kodlandı", flush=True)
        else:
            for i, doc in enumerate(docs, 1):
                arr = encode_document(tokenizer, doc)
                arr.tofile(f)
                n_tokens += len(arr)
                if i % 20_000 == 0:
                    print(f"  {label}: {i:,}/{len(docs):,} belge kodlandı", flush=True)
    print(f"  {label}: {n_tokens:,} token ({time.time() - start:.1f} sn)", flush=True)
    return n_tokens


# --------------------------------------------------------------------------
# Ana boru hattı
# --------------------------------------------------------------------------
def collect_documents(
    source: str,
    max_mb: float | None,
    file_path: str | Path | None = None,
    cache_dir: str | Path = "veri/onbellek",
    sentences_per_doc: int = 8,
    skip_mb: float = 0,
) -> list[str]:
    """Kaynaktan belgeleri okur, temizler, tekrarları atar; en fazla max_mb kadar.

    skip_mb: baştaki bu kadar MB'lık belgeyi atlar. Aynı kaynaktan ikinci bir
    eğitim yaparken modelin daha önce görmediği metni almak için kullanılır
    (ör. ilk eğitim 300 MB aldıysa ikincisinde --atla-mb 300).
    """
    if source == "ud":
        raw: Iterable[str] = ud_documents(cache_dir, sentences_per_doc)
    elif source == "wikipedia":
        raw = wikipedia_documents()
    elif source == "dosya":
        if file_path is None:
            raise SystemExit("Hata: --kaynak dosya için --dosya yol.txt vermelisiniz")
        raw = file_documents(file_path)
    else:
        raise SystemExit(f"Hata: bilinmeyen kaynak '{source}' (ud, wikipedia, dosya)")

    limit = None if max_mb is None else int(max_mb * 1024 * 1024)
    docs: list[str] = []
    n_bytes = 0
    skip = int(skip_mb * 1024 * 1024)
    skipped = 0
    cleaned = (clean_document(d) for d in raw)
    for doc in deduplicate(d for d in cleaned if d):
        if skipped < skip:
            skipped += len(doc.encode("utf-8"))
            continue
        docs.append(doc)
        n_bytes += len(doc.encode("utf-8"))
        if len(docs) % 50_000 == 0:
            print(f"  {len(docs):,} belge, {n_bytes / 1e6:.1f} MB", flush=True)
        if limit is not None and n_bytes >= limit:
            print(f"  --max-mb sınırına ({max_mb} MB) ulaşıldı, okuma durduruldu", flush=True)
            break
    return docs


def split_documents(docs: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Belgeleri karıştırıp eğitim/doğrulama olarak ayırır.

    Neden belge bazında? Token dizisini ortadan kesseydik aynı makalenin
    yarısı eğitimde, yarısı doğrulamada olurdu; model doğrulama metnini
    "kısmen görmüş" olurdu. Belge bazında ayırınca doğrulama gerçekten
    görülmemiş metin olur. Tohum (seed) sayesinde her çalıştırmada aynı ayrım.
    """
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    n_val = int(round(len(docs) * val_ratio))
    if len(docs) >= 2:
        n_val = min(max(n_val, 1), len(docs) - 1)  # iki tarafta da en az bir belge
    return docs[n_val:], docs[:n_val]


def sample_for_tokenizer(docs: list[str], max_mb: float, seed: int) -> list[str]:
    """Tokenizer eğitimi için en fazla max_mb'lık rastgele belge örneği seçer.

    BPE eğitimi saf Python'da yavaştır. Sık geçen parçaları (ekler, yaygın
    kelimeler) öğrenmek için birkaç on MB metin yeterlidir; 1 GB'ın tamamını
    kullanmak çok az fark yaratır ama saatler sürer.
    """
    limit = int(max_mb * 1024 * 1024)
    if sum(len(d.encode("utf-8")) for d in docs) <= limit:
        return docs  # veri zaten küçük, hepsini kullan
    order = list(range(len(docs)))
    random.Random(seed + 1).shuffle(order)
    sample, size = [], 0
    for i in order:
        sample.append(docs[i])
        size += len(docs[i].encode("utf-8"))
        if size >= limit:
            break
    return sample


def load_tokenizer(path: str | Path) -> BPETokenizer:
    """tokenizer.json'dan ya da bir model dosyasının (.pt) içinden tokenizer'ı yükler.

    Model dosyası tokenizer'ı da içinde taşıdığı için, kayıtlı bir modelin
    üstüne eğitirken doğrudan o model.pt dosyasını vermek yeterlidir.
    """
    path = Path(path)
    if path.suffix == ".pt":
        import torch  # sadece gerektiğinde; veri hazırlama torch'suz da çalışsın

        from .checkpoint import tokenizer_from_dict

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return tokenizer_from_dict(checkpoint["tokenizer"])
    return BPETokenizer.load(path)


def prepare(
    source: str,
    out_dir: str | Path,
    vocab_size: int = 8192,
    val_ratio: float = 0.02,
    seed: int = 42,
    max_mb: float | None = 200,
    file_path: str | Path | None = None,
    cache_dir: str | Path = "veri/onbellek",
    tokenizer_mb: float = 20,
    sentences_per_doc: int = 8,
    workers: int | None = None,
    tokenizer_path: str | Path | None = None,
    skip_mb: float = 0,
) -> dict:
    """Tüm boru hattını çalıştırır ve meta bilgisini (sözlük olarak) döner."""
    if vocab_size > 65536:
        raise SystemExit("Hata: uint16 en fazla 65536 farklı token tutabilir; --sozluk ≤ 65536 olmalı")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count() or 1
    t0 = time.time()

    print(f"[1/4] Belgeler okunuyor ve temizleniyor (kaynak: {source}) ...", flush=True)
    docs = collect_documents(source, max_mb, file_path, cache_dir, sentences_per_doc, skip_mb)
    if len(docs) < 2:
        raise SystemExit("Hata: temizlikten sonra en az 2 belge kalmalı (eğitim + doğrulama)")
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"  toplam {len(docs):,} belge, {total_bytes / 1e6:.2f} MB", flush=True)

    train_docs, val_docs = split_documents(docs, val_ratio, seed)
    del docs
    print(f"  eğitim: {len(train_docs):,} belge, doğrulama: {len(val_docs):,} belge", flush=True)

    if tokenizer_path is not None:
        # Var olan bir modelin üstüne eğiteceksek tokenizer DEĞİŞMEMELİ: model
        # "ev" kelimesini örneğin 812 numarasıyla öğrendi. Yeni bir tokenizer
        # aynı parçaya başka numara verirse model öğrendiklerini kullanamaz.
        # Bu yüzden yeni veriyi eski tokenizer'la kodluyoruz.
        print(f"[2/4] Var olan tokenizer kullanılıyor: {tokenizer_path}", flush=True)
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        # Tokenizer'ı SADECE eğitim belgeleriyle eğitiyoruz; doğrulama verisi
        # hiçbir aşamada "görülmemiş" kalmalı.
        print(f"[2/4] Tokenizer eğitiliyor (sözlük: {vocab_size}) ...", flush=True)
        tok_docs = sample_for_tokenizer(train_docs, tokenizer_mb, seed)
        if len(tok_docs) < len(train_docs):
            print(f"  büyük veri: tokenizer {len(tok_docs):,} belgelik örnekle eğitilecek", flush=True)
        t1 = time.time()
        tokenizer = BPETokenizer()
        tokenizer.train(tok_docs, vocab_size, verbose=True)
        print(f"  {tokenizer.vocab_size} token öğrenildi ({time.time() - t1:.1f} sn)", flush=True)
        if tokenizer.vocab_size < vocab_size:
            print("  not: veri küçük olduğu için istenen sözlük boyutuna ulaşılamadı", flush=True)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    print("[3/4] Belgeler kodlanıyor ...", flush=True)
    train_tokens = encode_to_file(train_docs, tokenizer, tokenizer_path, out_dir / "train.bin", workers, "eğitim")
    val_tokens = encode_to_file(val_docs, tokenizer, tokenizer_path, out_dir / "val.bin", workers, "doğrulama")

    print("[4/4] meta.json yazılıyor ...", flush=True)
    train_bytes = sum(len(d.encode("utf-8")) for d in train_docs)
    val_bytes = sum(len(d.encode("utf-8")) for d in val_docs)
    if source == "ud":
        license_info: dict | str = {f"UD_Turkish-{k}": v for k, v in UD_TREEBANKS.items()}
    elif source == "wikipedia":
        license_info = WIKIPEDIA_LICENSE
    else:
        license_info = "kullanıcı dosyası (lisansı kullanıcıya ait)"
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "kaynak": source,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        # Belge metinlerinin UTF-8 byte sayısı (<|son|> hariç). Bits-per-byte
        # hesabında işe yarar:  bpb = (kayıp / ln 2) * token_sayısı / byte_sayısı
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "bytes_per_token": round(train_bytes / max(train_tokens, 1), 4),
        "dtype": "uint16",
        "eot_id": tokenizer.special_id(END_OF_TEXT),
        "istenen_sozluk": vocab_size,
        "dogrulama_orani": val_ratio,
        "tohum": seed,
        "max_mb": max_mb,
        "lisans": license_info,
    }
    if source == "dosya":
        meta["dosya"] = str(file_path)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Bitti ({time.time() - t0:.1f} sn). {train_tokens:,} eğitim + {val_tokens:,} doğrulama tokeni "
        f"-> {out_dir}/  (token başına ~{meta['bytes_per_token']} byte)",
        flush=True,
    )
    return meta


# --------------------------------------------------------------------------
# Eğitim sırasında veri okuma
# --------------------------------------------------------------------------
class TokenDataset:
    """prepare() çıktısını okur ve eğitim için rastgele (x, y) parçaları verir.

    Dil modeli "bir sonraki tokeni tahmin et" görevini öğrenir. Bu yüzden
    hedef (y), girdinin (x) bir adım kaydırılmış halidir:
        tokenler: [5, 9, 2, 7, 3]
        x       : [5, 9, 2, 7]
        y       : [9, 2, 7, 3]   -> x[i]'den sonra y[i] gelmeli
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} bulunamadı. Önce veriyi hazırlayın: python -m turkce_gpt veri --help"
            )
        self.meta: dict = json.loads(meta_path.read_text(encoding="utf-8"))
        self.vocab_size: int = self.meta["vocab_size"]
        # mode="r": salt okunur. Dosya RAM'e yüklenmez, gerektikçe diskten okunur.
        self.data = {
            split: np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
            for split in ("train", "val")
        }

    def num_tokens(self, split: str) -> int:
        return len(self.data[split])

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: "str | torch.device" = "cpu"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """batch_size tane rastgele başlangıç noktasından block_size uzunluğunda parça keser."""
        import torch

        if split not in self.data:
            raise ValueError(f"split 'train' ya da 'val' olmalı, '{split}' verildi")
        data = self.data[split]
        if len(data) <= block_size:
            raise ValueError(
                f"'{split}' verisi ({len(data)} token) block_size={block_size} için çok kısa"
            )
        # Başlangıç i en fazla len - block_size - 1 olabilir; çünkü y, i+block_size'a kadar uzanır.
        ix = torch.randint(len(data) - block_size, (batch_size,))
        # uint16 -> int64: PyTorch'un embedding katmanı ve kayıp fonksiyonu int64 bekler.
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
        device = torch.device(device)
        if device.type == "cuda":
            # pin_memory: veriyi "sabitlenmiş" RAM'e koyar; GPU'ya kopyalama daha hızlı olur.
            # non_blocking=True: kopyalama sürerken işlemci bir sonraki işe geçebilir.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


# --------------------------------------------------------------------------
# Komut satırı
# --------------------------------------------------------------------------
def cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m turkce_gpt veri",
        description=(
            "Metni indirir/okur, temizler, BPE tokenizer eğitir ve token dosyalarını "
            "(train.bin, val.bin, tokenizer.json, meta.json) yazar."
        ),
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="bu yardım mesajını gösterir ve çıkar")
    parser.add_argument(
        "--kaynak", choices=["ud", "wikipedia", "dosya"], default="ud",
        help="metin kaynağı: ud (UD Türkçe treebank'leri, GitHub'dan), "
             "wikipedia (HuggingFace, 'datasets' gerekir), dosya (yerel .txt) (varsayılan: %(default)s)",
    )
    parser.add_argument("--dosya", help="--kaynak dosya için .txt yolu (boş satırla ayrılan bloklar belge sayılır)")
    parser.add_argument("--max-mb", type=float, default=200, help="en fazla bu kadar MB temiz metin kullan (varsayılan: %(default)s)")
    parser.add_argument("--cikti", default="veri/hazir", help="çıktı klasörü (varsayılan: %(default)s)")
    parser.add_argument("--sozluk", type=int, default=8192, help="tokenizer sözlük boyutu (vocab_size) (varsayılan: %(default)s)")
    parser.add_argument("--dogrulama-orani", type=float, default=0.02, help="doğrulamaya ayrılan belge oranı (varsayılan: %(default)s)")
    parser.add_argument("--tohum", type=int, default=42, help="karıştırma için rastgelelik tohumu (varsayılan: %(default)s)")
    parser.add_argument("--onbellek", default="veri/onbellek", help="indirilen UD depolarının klasörü (varsayılan: %(default)s)")
    parser.add_argument("--tokenizer-mb", type=float, default=20,
                        help="tokenizer en fazla bu kadar MB'lık örnekle eğitilir (BPE yavaştır) (varsayılan: %(default)s)")
    parser.add_argument("--cumle-sayisi", type=int, default=8,
                        help="ud kaynağında bir belgede birleştirilecek ardışık cümle sayısı (varsayılan: %(default)s)")
    parser.add_argument("--atla-mb", type=float, default=0,
                        help="kaynağın baştaki bu kadar MB'ını atla; aynı kaynaktan yeni (görülmemiş) metin almak için "
                             "(varsayılan: %(default)s)")
    parser.add_argument("--tokenizer", default=None,
                        help="yeni tokenizer eğitmek yerine bunu kullan: tokenizer.json ya da model.pt. "
                             "Kayıtlı bir modelin üstüne yeni veriyle eğitirken gereklidir (--sozluk yok sayılır)")
    parser.add_argument("--isci", type=int, default=None,
                        help="paralel kodlama için süreç sayısı (varsayılan: çekirdek sayısı)")
    args = parser.parse_args(argv)

    if args.kaynak == "dosya" and not args.dosya:
        parser.error("--kaynak dosya için --dosya yol.txt gerekli")

    prepare(
        source=args.kaynak,
        out_dir=args.cikti,
        vocab_size=args.sozluk,
        val_ratio=args.dogrulama_orani,
        seed=args.tohum,
        max_mb=args.max_mb,
        file_path=args.dosya,
        cache_dir=args.onbellek,
        tokenizer_mb=args.tokenizer_mb,
        sentences_per_doc=args.cumle_sayisi,
        workers=args.isci,
        tokenizer_path=args.tokenizer,
        skip_mb=args.atla_mb,
    )


if __name__ == "__main__":
    cli(sys.argv[1:])
