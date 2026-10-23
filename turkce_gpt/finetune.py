"""
İnce ayar (fine-tuning): ön eğitimli modeli sohbet etmeye alıştırır.

Komut:
    python -m turkce_gpt ince-ayar --model modeller/mini/model.pt \\
        --veri veri/sohbet.jsonl --cikti modeller/mini-sohbet

Ön eğitimde model Türkçeyi, kelimeleri, dilbilgisini öğrendi; ama "soru
sorulunca cevap ver, sonra sus" davranışını bilmiyor. İnce ayarda modele az
sayıda soru-cevap örneği gösteririz (bkz. chat.py'deki şablon).

Ön eğitimden farkları:
    - Veri çok daha az (yüzlerce örnek) ve her örnek ayrı bir konuşma.
    - Öğrenme oranı çok daha küçük (1e-4): model zaten iyi bir noktada; büyük
      adımlar ön eğitimde öğrendiklerini unutturur ("catastrophic forgetting").
    - Loss sadece asistan cevapları üzerinden hesaplanır (label maskesi).

Veri biçimi (her satır bir JSON):
    {"mesajlar": [{"role": "kullanici", "content": "..."}, {"role": "asistan", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .chat import IGNORE_INDEX, encode_conversation
from .checkpoint import load_checkpoint, save_checkpoint
from .tokenizer import END_OF_TEXT, BPETokenizer
from .train import append_log, format_duration, get_lr, resolve_device, setup_precision

Example = tuple[list[int], list[int]]  # (ids, labels)


def load_conversations(path: str | Path) -> list[list[dict]]:
    """jsonl dosyasından konuşmaları okur. Boş satırları atlar."""
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line)["mesajlar"])
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{line_no} satırı okunamadı: {e}") from e
    return conversations


def build_examples(tokenizer: BPETokenizer, conversations: list[list[dict]], block_size: int) -> list[Example]:
    """Konuşmaları (ids, labels) çiftlerine çevirir ve bağlama sığacak şekilde keser.

    Model en fazla block_size token görür. Girdi ids[:-1], hedef labels[1:]
    olacağı için bir örnek en fazla block_size + 1 token olabilir. Kesildikten
    sonra hiç öğrenilecek token kalmayan örnekleri atarız (loss NaN olurdu).
    """
    examples = []
    skipped = 0
    for messages in conversations:
        ids, labels = encode_conversation(tokenizer, messages)
        ids, labels = ids[: block_size + 1], labels[: block_size + 1]
        if all(label == IGNORE_INDEX for label in labels[1:]):
            skipped += 1
            continue
        examples.append((ids, labels))
    if skipped:
        print(f"UYARI: {skipped} konuşma bağlama sığmadığı için atlandı.")
    return examples


def collate(examples: list[Example], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Farklı uzunluktaki örnekleri tek bir tensöre koyar (padding).

    Bir yığındaki tüm satırlar aynı uzunlukta olmalı. Kısa örneklerin sonunu
    "dolgu" (pad) tokeniyle doldururuz. Dolgu konumlarının etiketi -100'dür,
    yani loss'a katkı vermezler. Model nedensel (causal) olduğu için gerçek
    tokenler sağdaki dolguyu zaten göremez; dolgu sonuçları değiştirmez.

    Kaydırma burada yapılır: x = ids[:-1], y = labels[1:]
    (her konumda bir SONRAKİ tokeni tahmin ediyoruz).
    """
    max_len = max(len(ids) for ids, _ in examples) - 1
    x = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labels) in enumerate(examples):
        n = len(ids) - 1
        x[row, :n] = torch.tensor(ids[:-1])
        y[row, :n] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_examples(model: torch.nn.Module, examples: list[Example], batch_size: int,
                      pad_id: int, device: str, ctx: object) -> float:
    """Örneklerin tamamında, öğrenilen token başına ortalama loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(examples), batch_size):
        x, y = collate(examples[i:i + batch_size], pad_id, device)
        with ctx:
            _, loss = model(x, y)
        # loss, bu yığındaki öğrenilen tokenlerin ortalaması; token sayısıyla
        # ağırlıklandırıp topluyoruz ki tüm veri üzerinden doğru ortalama çıksın.
        n = int((y != IGNORE_INDEX).sum())
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def finetune(args: argparse.Namespace) -> None:
    device = resolve_device(args.cihaz)
    random.seed(args.tohum)
    torch.manual_seed(args.tohum)

    model, tokenizer, info = load_checkpoint(args.model, device=device)
    model.train()
    block_size = model.config.block_size
    pad_id = tokenizer.special_id(END_OF_TEXT)  # dolgu için herhangi bir token olur; etiketi zaten -100

    conversations = load_conversations(args.veri)
    examples = build_examples(tokenizer, conversations, block_size)
    if not examples:
        raise SystemExit("Eğitilecek örnek yok: veri dosyasını kontrol edin.")

    # Örneklerin bir kısmını doğrulama için ayır. Çok az örnek varsa (ör. test)
    # ayırmak anlamsız; o zaman eğitim verisinin kendisinde ölçeriz.
    random.shuffle(examples)
    n_val = int(len(examples) * args.dogrulama_orani)
    if n_val >= 2:
        val_examples, train_examples = examples[:n_val], examples[n_val:]
    else:
        val_examples, train_examples = examples, examples
        print("Not: örnek sayısı az, doğrulama ayrı tutulmadı (loss eğitim verisinde ölçülecek).")

    out_dir = Path(args.cikti)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model.pt"
    log_path = out_dir / "log.csv"
    if log_path.exists():
        log_path.unlink()

    # Ön eğitimdeki gibi AdamW; ince ayarda weight decay genelde küçük tutulur.
    optimizer = model.configure_optimizer(weight_decay=args.agirlik_azalmasi, lr=args.lr, betas=(0.9, 0.95))
    ctx, scaler = setup_precision(device)
    min_lr = args.lr / 10

    print("=" * 70)
    print(f"Ön eğitimli model : {args.model} (adım {info.get('step', '?')}, val loss {info.get('val_loss', float('nan')):.4f})")
    print(f"Konuşma sayısı    : {len(train_examples)} eğitim, {len(val_examples)} doğrulama")
    print(f"Adım              : {args.adim}, yığın {args.batch}, lr {args.lr:g}")
    print("=" * 70)

    best_val = evaluate_examples(model, val_examples, args.batch, pad_id, device, ctx)
    print(f"Başlangıç cevap loss'u: {best_val:.4f}")

    t_start = time.time()
    order: list[int] = []
    for step in range(args.adim):
        lr = get_lr(step, args.lr, min_lr, args.isinma, args.adim)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Örnekleri karışık sırayla, hepsini tüketene kadar (bir "epoch") dolaş.
        if len(order) < args.batch:
            order += random.sample(range(len(train_examples)), len(train_examples))
        batch_idx, order = order[:args.batch], order[args.batch:]
        x, y = collate([train_examples[i] for i in batch_idx], pad_id, device)

        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # bkz. train.py
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        done = step + 1
        if done % args.log_araligi == 0 or done == args.adim:
            elapsed = time.time() - t_start
            eta = elapsed / done * (args.adim - done)
            print(f"adım {done:>5}/{args.adim} | loss {loss.item():.4f} | lr {lr:.2e} | kalan ~{format_duration(eta)}")

        if done % args.degerlendirme_araligi == 0 or done == args.adim:
            val_loss = evaluate_examples(model, val_examples, args.batch, pad_id, device, ctx)
            improved = val_loss < best_val
            print(f"DEĞERLENDİRME adım {done}: doğrulama loss {val_loss:.4f} | "
                  f"perplexity {math.exp(min(val_loss, 50)):.1f}{'  <- en iyi, kaydedildi' if improved else ''}")
            append_log(log_path, {"adim": done, "train_loss": round(loss.item(), 5),
                                  "val_loss": round(val_loss, 5), "lr": f"{lr:.3e}",
                                  "sure": round(time.time() - t_start, 1)})
            # İlk değerlendirmede de kaydediyoruz ki çıktı klasöründe mutlaka bir model olsun.
            if improved or not best_path.exists():
                best_val = min(best_val, val_loss)
                save_checkpoint(best_path, model, tokenizer, {
                    "step": done, "val_loss": val_loss, "tur": "sohbet",
                    "temel_model": str(args.model), "veri": str(args.veri),
                })

    print(f"İnce ayar bitti. En iyi doğrulama loss'u: {best_val:.4f} -> {best_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m turkce_gpt ince-ayar",
        description="Ön eğitimli modeli soru-cevap örnekleriyle sohbete alıştırır.",
    )
    p.add_argument("--model", default="modeller/mini/model.pt", help="ön eğitimli model dosyası")
    p.add_argument("--veri", default="veri/sohbet.jsonl", help="her satırı {\"mesajlar\": [...]} olan jsonl dosyası")
    p.add_argument("--cikti", default="modeller/mini-sohbet", help="ince ayarlı modelin yazılacağı klasör")
    p.add_argument("--adim", type=int, default=300, help="toplam optimizer adımı")
    p.add_argument("--batch", type=int, default=16, help="yığındaki konuşma sayısı")
    p.add_argument("--lr", type=float, default=1e-4, help="en yüksek öğrenme oranı")
    p.add_argument("--isinma", type=int, default=20, help="ısınma adımı sayısı")
    p.add_argument("--agirlik-azalmasi", type=float, default=0.0, help="AdamW weight decay katsayısı")
    p.add_argument("--dogrulama-orani", type=float, default=0.1, help="doğrulamaya ayrılacak konuşma oranı")
    p.add_argument("--degerlendirme-araligi", type=int, default=50, help="kaç adımda bir doğrulama loss'u ölçülsün")
    p.add_argument("--log-araligi", type=int, default=10, help="kaç adımda bir ilerleme yazılsın")
    p.add_argument("--cihaz", default="auto", help="auto, cpu, cuda veya mps")
    p.add_argument("--tohum", type=int, default=1337, help="rastgelelik tohumu")
    return p


def cli(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    finetune(args)
