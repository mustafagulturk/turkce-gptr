"""
Değerlendirme: eğitilmiş modelin doğrulama (val) verisindeki başarısını ölçer.

Komut:
    python -m turkce_gpt degerlendir --model modeller/mini/model.pt --veri veri/hazir

Üç ölçü yazdırılır:

1) Val loss (nats/token)
   Doğru tokene verilen olasılığın eksi logaritmasının ortalaması.
   Düşük = iyi. Doğal logaritma kullanıldığı için birimi "nat"tır.

2) Perplexity (karmaşıklık) = e^loss
   Sezgisel anlamı: model her tokende ortalama "kaç seçenek arasında
   kararsız" kalıyor. Perplexity 20 ise model, 20 eşit olasılıklı seçenek
   arasından rastgele seçen biri kadar şaşkın demektir. Rastgele bir model
   için perplexity sözlük boyutuna eşittir (ör. 8192).

3) Bits-per-byte (byte başına bit, bpb)
   Loss ve perplexity TOKENİZER'A BAĞLIDIR: büyük sözlüklü bir tokenizer
   metni daha az tokene böler, token başına loss artar ama bu modelin daha
   kötü olduğu anlamına gelmez. Farklı tokenizer'lı modelleri karşılaştırmak
   için metnin kendisine (byte'larına) göre ölçeriz:

       bpb = (toplam loss, nat cinsinden) / ln(2) / (toplam byte sayısı)

   ln(2)'ye bölmek nat'ı bit'e çevirir. Sonuç: modeli bir sıkıştırma
   programı gibi kullansaydık metnin her byte'ını ortalama kaç bitle
   saklayabilirdik. Düz UTF-8 metin byte başına 8 bit kullanır; iyi bir dil
   modeli 1 bitin altına inebilir. (Türkçe karakterler ç, ğ, ş UTF-8'de 2 byte'tır.)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .model import GPT
from .tokenizer import BPETokenizer
from .train import resolve_device, setup_precision


def bits_per_byte(total_nats: float, total_bytes: int) -> float:
    """Toplam loss'u (nat) byte başına bite çevirir."""
    return total_nats / math.log(2) / total_bytes


def token_byte_lengths(tokenizer: BPETokenizer) -> np.ndarray:
    """Her tokenin metinde kaç byte'a karşılık geldiği (token numarası -> byte sayısı).

    Özel tokenler ("<|son|>" gibi) gerçek metnin parçası değildir; belge
    sınırını gösteren tek bir ayraç karakteri gibi 1 byte sayıyoruz.
    """
    lengths = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    for i, b in tokenizer.vocab.items():
        lengths[i] = len(b)
    for i in tokenizer.special_tokens.values():
        lengths[i] = 1
    return lengths


@torch.no_grad()
def evaluate_tokens(
    model: GPT,
    tokens: np.ndarray,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    max_tokens: int | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Token dizisini baştan sona, üst üste binmeyen pencerelerle tarayıp ölçer.

    Eğitimdeki rastgele yığınlar yerine verinin tamamını (veya ilk max_tokens
    kadarını) sırayla geziyoruz; böylece sonuç her çalıştırmada aynıdır.
    Her pencere block_size + 1 token: ilk block_size'ı girdi, kaydırılmışı hedef.
    """
    model.eval()
    ctx, _ = setup_precision(device)
    block = model.config.block_size
    if max_tokens is not None:
        tokens = tokens[: max_tokens + 1]
    n_windows = (len(tokens) - 1) // block
    if n_windows == 0:
        raise ValueError(f"Değerlendirme için en az {block + 1} token gerekli.")
    byte_len = token_byte_lengths(tokenizer)

    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    for start in range(0, n_windows, batch_size):
        rows = range(start, min(start + batch_size, n_windows))
        chunk = np.stack([tokens[r * block: r * block + block + 1].astype(np.int64) for r in rows])
        x = torch.from_numpy(chunk[:, :-1]).to(device)
        y = torch.from_numpy(chunk[:, 1:]).to(device)
        with ctx:
            logits, _ = model(x)
        # reduction="sum": ortalama değil toplam loss; tüm veri bitince kendimiz böleceğiz.
        nats = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nats += nats.item()
        total_tokens += y.numel()
        # Tahmin edilen (hedef) tokenlerin metindeki byte uzunluğu.
        total_bytes += int(byte_len[chunk[:, 1:]].sum())

    loss = total_nats / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_byte": bits_per_byte(total_nats, total_bytes),
        "tokens": total_tokens,
        "bytes": total_bytes,
        "bytes_per_token": total_bytes / total_tokens,
    }


def cli(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="python -m turkce_gpt degerlendir",
        description="Modelin doğrulama verisindeki loss, perplexity ve bits-per-byte değerlerini ölçer.",
    )
    p.add_argument("--model", default="modeller/mini/model.pt", help="değerlendirilecek model dosyası")
    p.add_argument("--veri", default="veri/hazir", help="val.bin dosyasını içeren klasör")
    p.add_argument("--bolum", default="val", choices=["val", "train"], help="hangi veri bölümü ölçülsün")
    p.add_argument("--batch", type=int, default=8, help="aynı anda işlenecek pencere sayısı")
    p.add_argument("--max-token", type=int, default=500_000, help="en fazla kaç token ölçülsün (0: hepsi)")
    p.add_argument("--cihaz", default="auto", help="auto, cpu, cuda veya mps")
    args = p.parse_args(argv)

    device = resolve_device(args.cihaz)
    model, tokenizer, info = load_checkpoint(args.model, device=device)
    # .bin dosyası: uint16 token numaralarının düz dizisi (bkz. data.py).
    # memmap dosyayı belleğe tamamen yüklemeden okur.
    tokens = np.memmap(Path(args.veri) / f"{args.bolum}.bin", dtype=np.uint16, mode="r")
    if int(tokens.max(initial=0)) >= model.config.vocab_size:
        raise SystemExit("Veri, modelin sözlüğünde olmayan token numaraları içeriyor: "
                         "model başka bir tokenizer'la eğitilmiş olabilir.")

    print(f"Model: {args.model} ({model.num_params() / 1e6:.2f} M parametre, eğitim adımı {info.get('step', '?')})")
    print(f"Veri : {args.veri}/{args.bolum}.bin ({len(tokens):,} token)")
    result = evaluate_tokens(model, tokens, tokenizer, args.batch, args.max_token or None, device)
    print("-" * 50)
    print(f"Ölçülen token        : {result['tokens']:,} ({result['bytes']:,} byte, "
          f"token başına {result['bytes_per_token']:.2f} byte)")
    print(f"Loss (nat/token)     : {result['loss']:.4f}")
    print(f"Perplexity           : {result['perplexity']:.2f}   (rastgele model: {model.config.vocab_size})")
    print(f"Bits-per-byte        : {result['bits_per_byte']:.4f}   (düz UTF-8: 8.0)")
    print("-" * 50)
