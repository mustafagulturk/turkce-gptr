"""
Eğitilmiş modelle konuşma: metin üretme (uret) ve sohbet (sohbet) komutları.

    python -m turkce_gpt uret   --model modeller/mini/model.pt --baslangic "Bir varmış"
    python -m turkce_gpt sohbet --model modeller/mini-sohbet/model.pt

Model bir seferde tek token üretir. Kullanıcı beklemesin diye her tokeni
üretildiği anda ekrana basarız (akış / "streaming"). Burada küçük ama önemli
bir sorun var: bir token tam bir harf olmak zorunda değildir!

    Türkçedeki 'ş' harfi UTF-8'de iki byte'tır: 0xC5 0x9F.
    Byte düzeyindeki BPE'de bu iki byte ayrı tokenlerde olabilir.
    İlk tokeni tek başına çözersek ekranda '�' görürüz.

Çözüm "artımlı çözme" (incremental decoding): tokenlerin byte'larını
biriktiririz ve sadece TAMAMLANMIŞ karakterleri ekrana yazarız. Yarım kalan
byte'lar bir sonraki tokeni bekler. Bunu `IncrementalDecoder` sınıfı yapar.
"""

from __future__ import annotations

import argparse
import codecs
import sys
from typing import Iterable

import torch

from turkce_gpt.chat import encode_prompt
from turkce_gpt.checkpoint import load_checkpoint
from turkce_gpt.tokenizer import END_OF_TEXT, SPECIAL_TOKENS, USER, BPETokenizer


# --------------------------------------------------------------- yardımcılar
class IncrementalDecoder:
    """Token numaralarını tek tek alıp sadece tam çözülen metni döndürür.

    Python'un `codecs` modülündeki artımlı UTF-8 çözücüyü kullanırız: ona byte
    verdikçe, tamamlanan karakterleri döndürür; yarım kalan çok byte'lı bir
    karakterin başını içinde saklar ve sonraki byte'ları bekler.

    Örnek ('ş' = 0xC5 0x9F, iki ayrı tokende geldiğini varsayalım):
        add(token_C5) -> ""    (henüz yarım, bir şey basma)
        add(token_9F) -> "ş"   (karakter tamamlandı)
    """

    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer = tokenizer
        # errors="replace": model gerçekten geçersiz bir byte dizisi üretirse
        # (ör. yarım karakterden sonra başka bir harf) program çökmesin, '�' bassın.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def add(self, token_id: int) -> str:
        """Bir tokeni ekler; ekrana basılmaya hazır (tamamlanmış) metni döndürür."""
        # tokenizer.vocab: token numarası -> o tokenin byte karşılığı.
        return self._decoder.decode(self.tokenizer.vocab[token_id])

    def flush(self) -> str:
        """Üretim bitti: içerde kalan yarım byte varsa onu da ('�' olarak) çıkarır."""
        return self._decoder.decode(b"", final=True)


def decode_stream(tokenizer: BPETokenizer, ids: Iterable[int]) -> list[str]:
    """Bir token dizisini akış halinde çözer ve her adımda basılacak parçaları döndürür.

    Parçaları birleştirince `tokenizer.decode(ids)` ile aynı metin çıkar; fark,
    hiçbir parçanın yarım karakter içermemesidir. (Testlerde kullanılıyor.)
    """
    decoder = IncrementalDecoder(tokenizer)
    pieces = [decoder.add(i) for i in ids]
    pieces.append(decoder.flush())
    return pieces


def pick_device(name: str = "auto") -> str:
    """'auto' ise en hızlı kullanılabilir cihazı seçer: NVIDIA GPU > Apple GPU > işlemci."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def strip_special_tokens(text: str) -> str:
    """Metinden "<|asistan|>" gibi özel token yazılarını siler.

    Modelin cevabını geçmişe eklemeden önce kullanırız: cevap metni bu
    işaretleri düz yazı olarak içerirse sonraki turlarda kafa karıştırabilir.
    """
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def _sampling_kwargs(args: argparse.Namespace) -> dict:
    """CLI seçeneklerini `generate_stream`'in beklediği adlara çevirir.

    top-k 0 ve top-p 1.0 "filtre yok" demektir; modele None olarak iletiriz.
    """
    return dict(
        temperature=args.sicaklik,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0.0 < args.top_p < 1.0 else None,
        repetition_penalty=args.tekrar_cezasi,
    )


def _add_common_args(parser: argparse.ArgumentParser, temperature: float, top_p: float,
                     repetition_penalty: float, seed: int | None) -> None:
    """'uret' ve 'sohbet' komutlarında ortak olan seçenekler."""
    parser.add_argument("--model", required=True, help="Eğitilmiş model dosyası (.pt)")
    parser.add_argument("--uzunluk", type=int, default=200,
                        help="En fazla kaç yeni token üretilsin (varsayılan: %(default)s)")
    parser.add_argument("--sicaklik", type=float, default=temperature,
                        help="Sıcaklık: düşükse model temkinli ve tekrarcı, yüksekse yaratıcı "
                             "ama dağınık olur; 0 her adımda en olası tokeni seçer "
                             "(varsayılan: %(default)s)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Sadece en olası k token arasından seç; 0 = sınır yok "
                             "(varsayılan: %(default)s)")
    parser.add_argument("--top-p", type=float, default=top_p,
                        help="Olasılıkları toplamı p'yi geçene kadarki en olası tokenler "
                             "arasından seç (nucleus); 1.0 = kapalı (varsayılan: %(default)s)")
    parser.add_argument("--tekrar-cezasi", type=float, default=repetition_penalty,
                        help="Daha önce geçen tokenlerin olasılığını düşürür; 1.0 = kapalı "
                             "(varsayılan: %(default)s)")
    parser.add_argument("--tohum", type=int, default=seed,
                        help="Rastgelelik tohumu; aynı tohum aynı çıktıyı verir "
                             "(varsayılan: %(default)s)")
    parser.add_argument("--cihaz", default="auto",
                        help="auto, cpu, cuda veya mps (varsayılan: %(default)s)")


# ------------------------------------------------------------------ uret
def build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m turkce_gpt uret",
        description="Verilen başlangıç metnini model ile devam ettirir ve akış halinde ekrana basar.",
    )
    parser.add_argument("--baslangic", default="Bir varmış",
                        help="Modelin devam ettireceği metin (varsayılan: %(default)r)")
    _add_common_args(parser, temperature=0.8, top_p=0.95, repetition_penalty=1.0, seed=42)
    return parser


def generate_cli(argv: list[str]) -> None:
    args = build_generate_parser().parse_args(argv)
    if args.tohum is not None:
        torch.manual_seed(args.tohum)  # örnekleme rastgeleliği tekrarlanabilir olsun

    device = pick_device(args.cihaz)
    model, tokenizer, _ = load_checkpoint(args.model, device=device)
    end_id = tokenizer.special_id(END_OF_TEXT)

    # Kullanıcının metnini kodla. Boşsa, "<|son|>" ile başlarız: model bunu
    # "yeni bir belge başlıyor" diye öğrendi (veri hazırlanırken belgeler
    # arasına bu token konuyor).
    ids = tokenizer.encode(args.baslangic, allow_special=False) or [end_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # şekil [1, T]

    print(args.baslangic, end="", flush=True)
    decoder = IncrementalDecoder(tokenizer)
    # Model "<|son|>" üretirse metin bitmiş demektir; orada dururuz.
    for token_id in model.generate_stream(idx, max_new_tokens=args.uzunluk,
                                          stop_ids={end_id}, **_sampling_kwargs(args)):
        # Sadece tamamlanmış karakterleri yaz; flush=True ile tampon beklemeden ekrana.
        print(decoder.add(token_id), end="", flush=True)
    print(decoder.flush())


# ---------------------------------------------------------------- sohbet
def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m turkce_gpt sohbet",
        description="Modelle etkileşimli sohbet. Komutlar: /temizle (geçmişi sil), /cikis (çık).",
    )
    # Sohbette biraz daha temkinli örnekleme ve hafif tekrar cezası daha tutarlı cevap verir.
    _add_common_args(parser, temperature=0.7, top_p=0.9, repetition_penalty=1.1, seed=None)
    return parser


def build_chat_ids(tokenizer: BPETokenizer, history: list[dict], max_prompt_tokens: int) -> list[int]:
    """Sohbet geçmişinden modele verilecek token dizisini kurar.

    Modelin bağlamı (block_size) sınırlıdır. Geçmiş sığmazsa EN ESKİ turları
    (bir kullanıcı + bir asistan mesajı) atarız; son kullanıcı mesajı her zaman
    kalır. Bu `history` listesini yerinde kısaltır, böylece sonraki turlarda da
    atılan mesajlar geri gelmez.
    """
    while True:
        # encode_prompt, chat.format_prompt ile AYNI şablonu
        # ("<|kullanici|>soru<|asistan|>cevap<|son|>...<|asistan|>") doğrudan
        # tokenlere çevirir. Özel tokenleri kendisi ekler, kullanıcı metnini ise
        # allow_special=False ile kodlar; böylece kullanıcı "<|asistan|>" yazarak
        # sahte bir tur ekleyemez (prompt injection koruması). Sondaki
        # "<|asistan|>" modele "şimdi cevap sırası sende" der.
        ids = encode_prompt(tokenizer, history)
        if len(ids) <= max_prompt_tokens or len(history) <= 1:
            break
        del history[:2]  # en eski (kullanici, asistan) çiftini at
    # Tek mesaj bile sığmıyorsa, son max_prompt_tokens tokeni tut (baştan kes).
    return ids[-max_prompt_tokens:]


def _read_user_line() -> str | None:
    """Kullanıcıdan bir satır okur; Ctrl+D / Ctrl+C gelirse None döndürür."""
    try:
        line = input("\nSen: ")
    except (EOFError, KeyboardInterrupt):
        return None
    # Girdi bir dosyadan/borudan geliyorsa (ör. Colab'da `printf ... |`) terminal
    # yazılanı göstermez; sohbet okunaklı olsun diye satırı kendimiz yazarız.
    if not sys.stdin.isatty():
        print(line)
    return line


def chat_cli(argv: list[str]) -> None:
    args = build_chat_parser().parse_args(argv)
    if args.tohum is not None:
        torch.manual_seed(args.tohum)

    device = pick_device(args.cihaz)
    model, tokenizer, _ = load_checkpoint(args.model, device=device)
    # Cevap "<|son|>" ile biter (ince ayarda böyle öğrendi). Model kafası karışıp
    # kullanıcı adına yeni bir tur ("<|kullanici|>") başlatmaya kalkarsa da dururuz.
    stop_ids = {tokenizer.special_id(END_OF_TEXT), tokenizer.special_id(USER)}

    # Bağlamın bir kısmını cevaba ayırıyoruz; kalanı geçmiş (istem) için.
    # Cevap payı bağlamın yarısını geçmesin ki geçmişe de yer kalsın.
    block_size = model.config.block_size
    reply_budget = min(args.uzunluk, block_size // 2)
    max_prompt_tokens = block_size - reply_budget

    print(f"Çınar sohbet ({device}). Komutlar: /temizle, /cikis")
    history: list[dict] = []

    while True:
        line = _read_user_line()
        if line is None:
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/cikis", "/çıkış"):
            break
        if text == "/temizle":
            history.clear()
            print("(Sohbet geçmişi silindi.)")
            continue

        history.append({"role": "kullanici", "content": text})
        ids = build_chat_ids(tokenizer, history, max_prompt_tokens)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("Çınar: ", end="", flush=True)
        decoder = IncrementalDecoder(tokenizer)
        reply_ids: list[int] = []
        try:
            for token_id in model.generate_stream(idx, max_new_tokens=args.uzunluk,
                                                  stop_ids=stop_ids, **_sampling_kwargs(args)):
                reply_ids.append(token_id)
                print(decoder.add(token_id), end="", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C uzun bir cevabı yarıda keser ama programı kapatmaz.
            print(" (kesildi)", end="")
        print(decoder.flush())

        # Cevabı geçmişe ekle ki model bir sonraki turda neyi konuştuğumuzu bilsin.
        reply = strip_special_tokens(tokenizer.decode(reply_ids)).strip()
        history.append({"role": "asistan", "content": reply})

    print("Görüşmek üzere!")
