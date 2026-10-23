"""
Byte düzeyinde BPE (Byte Pair Encoding) tokenizer'ı, sıfırdan.

Neden tokenizer gerekiyor?
    Sinir ağı metni değil sayıları anlar. Tokenizer metni küçük parçalara
    (token) böler ve her parçaya bir numara verir:
        "Merhaba dünya" -> [312, 1045, 87]  (örnek)
    Model bu numaralarla çalışır, sonra numaraları tekrar metne çeviririz.

BPE nasıl çalışır?
    1. Başlangıçta her byte (0-255) ayrı bir tokendir. Böylece hiçbir karakter
       "bilinmeyen" olmaz; ğ, ş, İ, emoji... hepsi byte'lardan oluşur.
    2. Eğitim metninde yan yana en sık geçen iki tokeni bul (ör. "l" + "a").
    3. Bu ikiliyi yeni bir tokene birleştir ("la") ve metinde yerine koy.
    4. İstenen sözlük boyutuna ulaşana kadar 2-3'ü tekrarla.
    Sonuçta "ler", "lar", "yor", " bir" gibi Türkçede çok geçen parçalar tek
    token olur. Türkçe eklemeli bir dil olduğu için bu yöntem çok işe yarar:
    "evlerimizden" -> " ev" + "lerimiz" + "den" gibi parçalanabilir.

GPT-2, GPT-4 ve Llama da aynı fikri kullanır.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Metni BPE'den önce kelime benzeri parçalara ayıran düzenli ifade.
# Birleştirmeler bu parçaların SINIRINI aşamaz; böylece "kitap." içindeki
# nokta ile harfler aynı tokende birleşmez.
#   ' ?[^\W\d_]+'  -> başında isteğe bağlı boşluk olan harf dizisi (" kitap")
#   ' ?\d{1,3}'    -> en fazla 3 haneli sayı parçası (sayıları böler: "2026" -> "202","6")
#   ' ?[^\s\w]+'   -> noktalama/simge dizisi ("...", "!?")
#   '\s+(?!\S)'    -> satır sonundaki boşluklar
#   '\s+'          -> diğer boşluklar
# Python'un "re" modülünde \w Unicode harflerini (ç, ğ, ı, ö, ş, ü) zaten kapsar.
PRETOKEN_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+")

# Özel tokenler: metinde normal kelime olarak geçmeyecek işaretler.
END_OF_TEXT = "<|son|>"        # bir belgenin / cevabın bittiğini söyler
USER = "<|kullanici|>"         # sohbette kullanıcının mesajı başlıyor
ASSISTANT = "<|asistan|>"      # sohbette modelin cevabı başlıyor
SPECIAL_TOKENS = [END_OF_TEXT, USER, ASSISTANT]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """ids listesinde yan yana duran 'pair' ikililerini new_id ile değiştirir.

    Örnek: _merge([5, 6, 5, 6, 7], (5, 6), 99) -> [99, 99, 7]
    """
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # merges: (sol_token, sag_token) -> yeni_token. Sırası önemlidir:
        # önce öğrenilen birleştirme, kodlamada da önce uygulanır.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: token numarası -> o tokenin byte karşılığı
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ eğitim
    def train(self, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        """Verilen metinlerden BPE birleştirmelerini öğrenir.

        Basit yöntem her adımda tüm metni baştan tarar ve çok yavaştır. Burada
        iki hızlandırma var:
          1. Aynı kelime binlerce kez geçer; her FARKLI kelimeyi bir kez
             tutup yanına kaç kez geçtiğini (frekans) yazarız.
          2. İkili sayımlarını (pair_counts) her adımda yeniden hesaplamak
             yerine, sadece birleştirmeden etkilenen kelimeleri güncelleriz.
          3. En sık ikiliyi bulmak için her adımda tüm ikilileri taramak yerine
             bir öncelik kuyruğu (heap) kullanırız. Sayımı değişen ikiliyi
             kuyruğa yeni değeriyle tekrar ekleriz; kuyruktan çıkan eski
             (artık geçersiz) kayıtları görmezden geliriz ("tembel silme").
        """
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        if n_merges < 0:
            raise ValueError("vocab_size en az 256 + özel token sayısı olmalı")

        # 1) Metni parçalara ayır ve her parçanın kaç kez geçtiğini say.
        word_freq: Counter[str] = Counter()
        for text in texts:
            word_freq.update(PRETOKEN_PATTERN.findall(text))

        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())

        # 2) Başlangıç ikili sayımları ve "hangi ikili hangi kelimelerde geçiyor" dizini.
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freqs[wi]
                pair_to_words[pair].add(wi)

        # heapq en KÜÇÜĞÜ verir; sayıyı eksi yazarak en büyüğü önce alırız.
        # Eşitlikte küçük numaralı ikili önce çıkar, böylece aynı veriyle her
        # eğitim aynı sonucu verir (tekrarlanabilirlik).
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        for step in range(n_merges):
            best = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count:
                    best = pair  # kayıt güncel, en sık ikili bu
                    break
                # Değilse bu kayıt eskimiş; at ve bir sonrakine bak.
            if best is None:
                break  # metin bitti, birleştirecek ikili kalmadı
            if pair_counts[best] < 2:
                break  # sadece bir kez geçen ikiliyi birleştirmenin faydası yok
            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Sadece bu ikiliyi içeren kelimeleri güncelle; sayımı değişen
            # ikilileri 'changed' kümesinde toplayıp sonra kuyruğa ekleriz.
            changed: set[tuple[int, int]] = set()
            for wi in pair_to_words.pop(best, set()):
                ids = words[wi]
                f = freqs[wi]
                # Eski ikililerin sayımını düş...
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] -= f
                    changed.add(pair)
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                # ...birleştir...
                ids = _merge(ids, best, new_id)
                words[wi] = ids
                # ...yeni ikilileri say.
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
                    pair_to_words[pair].add(wi)
                    changed.add(pair)
            for pair in changed:
                if pair in pair_counts:
                    heapq.heappush(heap, (-pair_counts[pair], pair))

            if verbose and (step + 1) % 500 == 0:
                print(f"  birleştirme {step + 1}/{n_merges}: {self.vocab[new_id]!r}")

        self._add_special_tokens()
        self._cache.clear()

    def _add_special_tokens(self) -> None:
        next_id = 256 + len(self.merges)
        self.special_tokens = {tok: next_id + i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, i in self.special_tokens.items():
            self.vocab[i] = tok.encode("utf-8")

    # --------------------------------------------------------------- kodlama
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Tek bir parçayı (ör. " kitaplar") öğrenilen birleştirmelerle tokenlere çevirir."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Mevcut ikililerden EN ÖNCE öğrenileni (en küçük numaralıyı) bul.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # uygulanabilecek birleştirme kalmadı
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 200_000:  # bellek şişmesin diye sınır
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Metni token numaralarına çevirir.

        allow_special=True ise metindeki "<|son|>" gibi ifadeler özel token
        olarak tanınır. Kullanıcıdan gelen metinde bunu kapalı tutarız; yoksa
        kullanıcı "<|asistan|>" yazarak modeli şaşırtabilir.
        """
        if allow_special and self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in PRETOKEN_PATTERN.findall(part):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Token numaralarını tekrar metne çevirir."""
        data = b"".join(self.vocab[i] for i in ids)
        # Model yarım bir Türkçe karakter (ör. 'ş'nin ilk byte'ı) üretebilir;
        # bu durumda hata vermek yerine yerine '�' koyarız.
        return data.decode("utf-8", errors="replace")

    def special_id(self, token: str) -> int:
        return self.special_tokens[token]

    # ------------------------------------------------------- kaydet / yükle
    def save(self, path: str | Path) -> None:
        data = {
            "tur": "turkce-gpt-bpe",
            "surum": 1,
            # JSON'da tuple anahtar olamaz; sıralı liste olarak saklıyoruz.
            "merges": [[a, b] for (a, b) in self.merges],
            "special_tokens": SPECIAL_TOKENS,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        # Birleştirmeleri öğrenildikleri sırayla yeniden oluştur.
        for i, (a, b) in enumerate(data["merges"]):
            new_id = 256 + i
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok._add_special_tokens()
        return tok
