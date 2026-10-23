"""
Ön eğitim (pretraining): modeli sıfırdan, düz metin üzerinde eğitir.

Komut:
    python -m turkce_gpt egit --veri veri/hazir --boyut mini --cikti modeller/mini

Modelin görevi çok basit: "metindeki bir sonraki tokeni tahmin et".
    girdi : "Bugün hava çok"
    hedef : "ün hava çok güzel"   (her konumda bir sonraki token)
Model her konum için sözlükteki tüm tokenlere bir olasılık verir. Doğru tokene
ne kadar düşük olasılık verdiyse o kadar ceza alır. Bu cezaya "kayıp" (loss)
denir ve cross-entropy ile ölçülür:
    loss = -log(doğru tokene verilen olasılık)
Doğru tokene 1.0 olasılık verse loss = 0; 0.01 verse loss = 4.6 olur.

Eğitim döngüsünün her adımı:
    1. Veriden rastgele bir yığın (batch) metin parçası al.
    2. İleri geçiş (forward): modelin tahminlerini ve loss'u hesapla.
    3. Geri yayılım (backward): her ağırlığın loss'u nasıl etkilediğini
       (gradyanı) hesapla.
    4. Optimizer adımı: ağırlıkları loss'u azaltacak yönde biraz değiştir.
Milyonlarca kez tekrarlanan bu küçük adımlar sonunda model dilin yapısını öğrenir.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import PRESETS, make_config
from .data import TokenDataset
from .model import GPT
from .tokenizer import END_OF_TEXT, BPETokenizer

LOG_COLUMNS = ["adim", "train_loss", "val_loss", "lr", "sure"]


# --------------------------------------------------------------------------
# Yardımcı fonksiyonlar (ince-ayar ve değerlendirme de bunları kullanır)
# --------------------------------------------------------------------------
def resolve_device(name: str) -> str:
    """'auto' ise en hızlı cihazı seçer: NVIDIA GPU (cuda) > Apple GPU (mps) > işlemci (cpu)."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Öğrenme oranı (learning rate) takvimi: doğrusal ısınma + kosinüs azalma.

    Öğrenme oranı, her adımda ağırlıkları ne kadar büyük değiştireceğimizdir.

        lr
        ^      ___
        |     /   ‾‾--__
        |    /          ‾-__
        |   /               ‾--___  min_lr
        +--+-------------------------> adım
          ısınma        kosinüs

    Isınma (warmup) neden gerekir?
        Eğitimin başında ağırlıklar rastgeledir ve gradyanlar büyük ve
        düzensizdir. AdamW'nin "gradyan ne kadar büyük oluyor" tahminleri de
        (ikinci moment) henüz birkaç örneğe dayanır, güvenilmezdir. Bu anda
        büyük adımlar atmak modeli kötü bir bölgeye fırlatabilir, loss
        patlayabilir (NaN). Bu yüzden lr'yi 0'a yakın bir değerden başlatıp
        ilk adımlarda yavaşça hedef değere çıkarırız.

    Kosinüs azalma neden?
        Eğitimin ortasında büyük adımlar hızlı ilerleme sağlar; sona doğru ise
        bir vadinin dibine yaklaşmışızdır ve büyük adımlar dibin etrafında
        zıplamamıza yol açar. lr'yi yavaşça küçültmek ince ayar yapmayı sağlar.
        Kosinüs eğrisi başta yavaş, ortada hızlı, sonda yine yavaş azalır.
    """
    # 1) Isınma: 0'dan max_lr'ye doğrusal artış. (step + 1) kullanıyoruz ki
    #    ilk adımın lr'si tam 0 olmasın (0 olsa o adım boşa giderdi).
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) Eğitim süresi bittiyse en düşük değerde kal.
    if step >= max_steps:
        return min_lr
    # 3) Kosinüs: ratio 0 -> 1 giderken coeff 1 -> 0 gider.
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def setup_precision(device: str) -> tuple[object, torch.amp.GradScaler]:
    """Karışık hassasiyet (mixed precision) ayarlarını döndürür: (autocast bağlamı, GradScaler).

    Normalde sayılar 32 bitlik (float32) tutulur. GPU'lar 16 bitlik sayılarla
    (bfloat16 / float16) çok daha hızlı hesap yapar ve yarı bellek kullanır.
    torch.autocast, matris çarpımı gibi işlemleri otomatik olarak 16 bitte,
    hassasiyet isteyen işlemleri (softmax, loss, normalizasyon) 32 bitte yapar.
    Ağırlıkların kendisi 32 bit kalır; böylece küçük güncellemeler kaybolmaz.

    bfloat16 vs float16:
        bfloat16 float32 ile aynı sayı ARALIĞINA sahiptir (sadece daha az
        hassas), yani taşma/sıfıra yuvarlanma sorunu yoktur. Yeni GPU'lar
        (A100, RTX 30xx+) destekler.
        float16'nın aralığı dardır: çok küçük gradyanlar 0'a yuvarlanabilir
        ("underflow"). GradScaler bunu çözmek için loss'u büyük bir sayıyla
        çarpar (gradyanlar da büyür), optimizer adımından önce geri böler.
        Colab'ın T4 GPU'su bfloat16 desteklemediği için orada bu yol kullanılır.

    İşlemcide (CPU) autocast kullanmıyoruz: orada hız kazancı azdır.
    """
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        return ctx, scaler
    return nullcontext(), torch.amp.GradScaler("cuda", enabled=False)


def format_count(n: float) -> str:
    """Büyük sayıları okunaklı yazar: 1_500_000 -> '1.5 M', 2_300_000_000 -> '2.3 B'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    return f"{int(n):,}"


def format_duration(seconds: float) -> str:
    """Saniyeyi okunaklı süreye çevirir: 3725 -> '1 sa 2 dk 5 sn'."""
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} sa {m} dk"
    if m:
        return f"{m} dk {s} sn"
    return f"{s} sn"


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: TokenDataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    ctx: object,
) -> dict[str, float]:
    """Eğitim ve doğrulama (val) verisinde ortalama loss'u ölçer.

    Neden val loss'a bakarız?
        Train loss, modelin GÖRDÜĞÜ veride ne kadar iyi olduğunu söyler. Model
        veriyi ezberleyerek de train loss'u düşürebilir. Val (doğrulama) verisi
        eğitimde hiç kullanılmaz; modelin yeni metinlerde ne kadar iyi
        olduğunu, yani GENELLEME yeteneğini ölçer.
          - İkisi birlikte düşüyorsa: model gerçekten öğreniyor.
          - Train düşüp val yükselmeye başladıysa: aşırı öğrenme (overfitting),
            yani ezber başladı. Bu yüzden en iyi modeli val loss'a göre seçeriz.

    Tek bir yığın şansa bağlı olarak kolay ya da zor olabilir; birkaç yığının
    ortalamasını alarak daha güvenilir bir ölçüm yaparız.
    """
    was_training = model.training
    model.eval()  # dropout kapalı: ölçüm tutarlı olsun
    result = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        result[split] = losses.mean().item()
    if was_training:
        model.train()
    return result


@torch.no_grad()
def generate_sample(model: GPT, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """Eğitimin gidişatını gözle görmek için kısa bir örnek metin üretir."""
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt) or [tokenizer.special_id(END_OF_TEXT)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = {tokenizer.special_id(END_OF_TEXT)}
    new_ids = list(model.generate_stream(idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, stop_ids=stop_ids))
    if was_training:
        model.train()
    return prompt + tokenizer.decode(new_ids)


def append_log(path: Path, row: dict) -> None:
    """log.csv'ye bir satır ekler (dosya yoksa başlık satırıyla oluşturur)."""
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------
# Eğitim
# --------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.cihaz)
    torch.manual_seed(args.tohum)
    np.random.seed(args.tohum)
    if device.startswith("cuda"):
        # TF32: Ampere ve sonrası GPU'larda float32 matris çarpımını çok daha
        # hızlı yapan, hassasiyeti hafifçe azaltan bir mod. Eğitim için yeterli.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.cikti)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model.pt"      # en iyi val loss'lu model (kullanım için)
    state_path = out_dir / "durum.pt"     # son durum + optimizer (--devam için)
    log_path = out_dir / "log.csv"

    # ---- Veri ve tokenizer
    dataset = TokenDataset(args.veri)
    tokenizer = BPETokenizer.load(Path(args.veri) / "tokenizer.json")
    if tokenizer.vocab_size != dataset.vocab_size:
        print(f"UYARI: tokenizer sözlüğü ({tokenizer.vocab_size}) ile veri sözlüğü "
              f"({dataset.vocab_size}) farklı. Veri yeniden hazırlanmalı olabilir.")

    # ---- Model ve optimizer
    start_step = 0
    best_val = float("inf")
    elapsed_before = 0.0  # önceki çalıştırmalarda geçen süre (--devam için)
    resume = args.devam and state_path.exists()
    if resume:
        model, _, extra = load_checkpoint(state_path, device=device)
        start_step = int(extra["step"])
        best_val = float(extra.get("best_val_loss", float("inf")))
        elapsed_before = float(extra.get("sure", 0.0))
        print(f"Kaldığı yerden devam ediliyor: adım {start_step}, en iyi val loss {best_val:.4f}")
    elif args.baslangic_modeli:
        # Kayıtlı bir modelin ağırlıklarıyla başla ("devam eğitimi"). --devam'dan
        # farkı: optimizer ve adım sayacı sıfırdan başlar, veri yeni olabilir.
        # Yeni bilgi öğrenirken eskisini unutmasın diye lr'yi ilk eğitimden
        # düşük tutmak (ör. 3e-4) iyi bir alışkanlıktır.
        model, start_tokenizer, _ = load_checkpoint(args.baslangic_modeli, device=device)
        if start_tokenizer.merges != tokenizer.merges:
            raise SystemExit(
                "Hata: başlangıç modelinin tokenizer'ı bu verininkiyle aynı değil. Veriyi "
                f"'veri --tokenizer {args.baslangic_modeli}' ile hazırlayın."
            )
        if args.dropout is not None:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = args.dropout
        print(f"Başlangıç ağırlıkları: {args.baslangic_modeli}")
        if log_path.exists():
            log_path.unlink()
    else:
        if args.devam:
            print(f"UYARI: {state_path} bulunamadı, eğitim sıfırdan başlıyor.")
        config = make_config(args.boyut, dataset.vocab_size, block_size=args.blok, dropout=args.dropout)
        model = GPT(config).to(device)
        if log_path.exists():
            log_path.unlink()  # sıfırdan eğitimde eski kaydı temizle
    model.train()
    config = model.config
    block_size = config.block_size

    # Neden AdamW?
    #   Düz SGD her ağırlığı gradyanla orantılı ve aynı lr ile günceller.
    #   Adam her ağırlık için iki "hafıza" tutar:
    #     - 1. moment (momentum): gradyanların hareketli ortalaması. Gürültülü
    #       gradyanları yumuşatır, tutarlı yönde hızlanmayı sağlar (beta1=0.9).
    #     - 2. moment: gradyan büyüklüğünün (karesinin) hareketli ortalaması.
    #       Adımı buna bölerek her ağırlığa kendi ölçeğinde bir adım attırır;
    #       nadir güncellenen ağırlıklar da yeterince öğrenir (beta2=0.95).
    #   "W" (decoupled weight decay): her adımda ağırlıkları sıfıra doğru biraz
    #   küçültür; aşırı büyük ağırlıkları ve ezberi engelleyen bir düzenlileştirme.
    #   Adam'daki klasik L2 cezasından farklı olarak 2. momente bölünmez, bu
    #   yüzden tüm ağırlıklara eşit etki eder. Transformer'lar için standart seçim.
    optimizer = model.configure_optimizer(weight_decay=args.agirlik_azalmasi, lr=args.lr, betas=(0.9, 0.95))
    if resume:
        optimizer.load_state_dict(extra["optimizer"])

    ctx, scaler = setup_precision(device)

    # torch.compile: modeli GPU için optimize edilmiş koda derler (ilk adım
    # yavaştır, sonra %20-50 hızlanma). Üretim için derlenmemiş halini tutuyoruz.
    raw_model = model
    if args.derle:
        print("Model derleniyor (torch.compile), ilk adım biraz sürebilir...")
        model = torch.compile(model)

    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    tokens_per_step = args.batch * args.birikim * block_size
    n_params = raw_model.num_params()

    print("=" * 70)
    print(f"Cihaz           : {device}")
    model_name = args.baslangic_modeli if (args.baslangic_modeli and not resume) else f"'{args.boyut}' boyutu"
    print(f"Model           : {model_name}, {format_count(n_params)} parametre "
          f"({config.n_layer} katman, {config.n_head} başlık, genişlik {config.n_embd}, bağlam {block_size})")
    print(f"Sözlük          : {config.vocab_size} token")
    print(f"Adım başına     : {args.batch} x {args.birikim} (birikim) x {block_size} = {tokens_per_step:,} token")
    print(f"Toplam          : {args.adim} adım ≈ {format_count(tokens_per_step * args.adim)} token")
    print(f"Öğrenme oranı   : {args.lr:g} -> {min_lr:g} (ısınma {args.isinma} adım)")
    print("=" * 70)

    if not resume:
        # Başlangıç kontrolü: rastgele bir model tüm tokenlere aşağı yukarı eşit
        # olasılık verir (1/V). O zaman loss ≈ -log(1/V) = ln(V) olmalı. Çok
        # farklıysa ilk ağırlıklarda (init) bir sorun vardır.
        losses = estimate_loss(model, dataset, args.degerlendirme_batch, args.batch, block_size, device, ctx)
        print(f"Başlangıç val loss: {losses['val']:.4f} "
              f"(rastgele tahminde beklenen ≈ ln({config.vocab_size}) = {math.log(config.vocab_size):.4f})")

    t_start = time.time()
    t_window = time.time()
    window_tokens = 0
    window_steps = 0
    sec_per_step = None  # hareketli ortalama; kalan süre tahmini için

    for step in range(start_step, args.adim):
        # Bu adımın öğrenme oranını takvimden alıp optimizer'a yazıyoruz.
        lr = get_lr(step, args.lr, min_lr, args.isinma, args.adim)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradyan biriktirme (gradient accumulation):
        #   Büyük yığınlar (batch) daha az gürültülü gradyan verir ama GPU
        #   belleğine sığmayabilir. Çözüm: yığını "birikim" sayıda küçük
        #   parçaya bölüp her birinin gradyanını sırayla hesaplamak. PyTorch
        #   .backward() çağrıldıkça gradyanları TOPLAR; optimizer adımını
        #   ancak hepsi bitince atarız. Loss'u parça sayısına bölüyoruz ki
        #   toplam, büyük yığının ORTALAMA loss'una eşit olsun.
        loss_accum = 0.0
        for _ in range(args.birikim):
            x, y = dataset.get_batch("train", args.batch, block_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.birikim
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradyan kırpma (gradient clipping):
        #   Bazen tek bir kötü yığın çok büyük bir gradyan üretir ve tek adımda
        #   modeli bozabilir ("gradyan patlaması"). Tüm gradyanları tek bir
        #   vektör gibi düşünüp uzunluğu (normu) 1.0'ı geçerse 1.0'a ölçekleriz.
        #   Yön aynı kalır, sadece adımın boyu sınırlanır.
        #   (fp16'da önce GradScaler'ın büyüttüğü gradyanları geri küçültmeliyiz.)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.kirpma).item()

        # Ağırlıkları güncelle. GradScaler, fp16'da gradyanlarda inf/NaN
        # görürse bu adımı atlar ve ölçeği küçültür.
        scaler.step(optimizer)
        scaler.update()
        # Gradyanları sıfırla; yoksa bir sonraki adımda üstüne eklenirler.
        optimizer.zero_grad(set_to_none=True)

        done = step + 1
        window_tokens += tokens_per_step
        window_steps += 1

        # ---- Ekrana ilerleme yaz
        if done % args.log_araligi == 0 or done == args.adim:
            if device.startswith("cuda"):
                torch.cuda.synchronize()  # GPU işini bitirsin ki süre doğru ölçülsün
            dt = time.time() - t_window
            tok_per_sec = window_tokens / max(dt, 1e-9)
            step_time = dt / window_steps
            # Kalan süre için adım süresinin yumuşatılmış ortalaması (ilk adımlar
            # derleme/ısınma yüzünden yavaş olabilir, ani değişimlere kapılmayalım).
            sec_per_step = step_time if sec_per_step is None else 0.7 * sec_per_step + 0.3 * step_time
            eta = sec_per_step * (args.adim - done)
            print(f"adım {done:>6}/{args.adim} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"grad norm {grad_norm:.2f} | {tok_per_sec:,.0f} token/sn | kalan ~{format_duration(eta)}")
            t_window = time.time()
            window_tokens = 0
            window_steps = 0

        # ---- Değerlendirme, kayıt ve örnek metin
        if done % args.degerlendirme_araligi == 0 or done == args.adim:
            losses = estimate_loss(model, dataset, args.degerlendirme_batch, args.batch, block_size, device, ctx)
            elapsed = elapsed_before + time.time() - t_start
            # Perplexity = e^loss. Sezgisel anlamı: model her adımda ortalama
            # "kaç token arasında kararsız kalıyor". 1 mükemmel; sözlük
            # boyutuna (V) eşitse model hiçbir şey bilmiyor demektir.
            val_ppl = math.exp(min(losses["val"], 50))
            improved = losses["val"] < best_val
            mark = "  <- en iyi, kaydedildi" if improved else ""
            print("-" * 70)
            print(f"DEĞERLENDİRME adım {done}: train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | val perplexity {val_ppl:.1f} | "
                  f"geçen süre {format_duration(elapsed)}{mark}")

            append_log(log_path, {
                "adim": done,
                "train_loss": round(losses["train"], 5),
                "val_loss": round(losses["val"], 5),
                "lr": f"{lr:.3e}",
                "sure": round(elapsed, 1),
            })

            info = {"step": done, "train_loss": losses["train"], "val_loss": losses["val"],
                    "tur": "on-egitim", "veri": str(args.veri)}
            if improved:
                best_val = losses["val"]
                save_checkpoint(best_path, raw_model, tokenizer, info)
            # Devam edebilmek için son durumu (optimizer dahil) her seferinde kaydet.
            save_checkpoint(state_path, raw_model, tokenizer,
                            {**info, "optimizer": optimizer.state_dict(), "best_val_loss": best_val,
                             "sure": elapsed})

            if args.ornek_uzunluk > 0:
                sample = generate_sample(raw_model, tokenizer, args.ornek_baslangic, args.ornek_uzunluk, device)
                print(f"Örnek metin: {sample!r}")
            print("-" * 70)
            # Değerlendirme süresi hız ölçümünü bozmasın.
            t_window = time.time()
            window_tokens = 0
            window_steps = 0

    print(f"Eğitim bitti. En iyi val loss: {best_val:.4f} -> {best_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m turkce_gpt egit",
        description="GPT modelini hazırlanmış token verisi üzerinde sıfırdan eğitir (ön eğitim).",
    )
    p.add_argument("--veri", default="veri/hazir", help="'veri' komutunun çıktı klasörü (train.bin, val.bin, tokenizer.json)")
    p.add_argument("--boyut", default="mini", choices=list(PRESETS), help="hazır model boyutu")
    p.add_argument("--cikti", default="modeller/mini", help="model.pt, durum.pt ve log.csv'nin yazılacağı klasör")
    p.add_argument("--adim", type=int, default=2000, help="toplam optimizer adımı")
    p.add_argument("--batch", type=int, default=32, help="bir ileri geçişteki örnek sayısı")
    p.add_argument("--birikim", type=int, default=1, help="gradyan biriktirme: kaç küçük yığın bir adım sayılır")
    p.add_argument("--lr", type=float, default=1e-3, help="en yüksek öğrenme oranı")
    p.add_argument("--min-lr", type=float, default=None, help="kosinüs sonunda varılacak öğrenme oranı (varsayılan lr/10)")
    p.add_argument("--isinma", type=int, default=100, help="ısınma (warmup) adımı sayısı")
    p.add_argument("--agirlik-azalmasi", type=float, default=0.1, help="AdamW weight decay katsayısı")
    p.add_argument("--kirpma", type=float, default=1.0, help="gradyan kırpma eşiği (norm)")
    p.add_argument("--dropout", type=float, default=None, help="dropout oranı (varsayılan boyuttaki değer)")
    p.add_argument("--blok", type=int, default=None, help="bağlam uzunluğunu (block_size) değiştir")
    p.add_argument("--degerlendirme-araligi", type=int, default=200, help="kaç adımda bir val loss ölçülsün")
    p.add_argument("--degerlendirme-batch", type=int, default=20, help="val loss ölçümünde kullanılacak yığın sayısı")
    p.add_argument("--log-araligi", type=int, default=10, help="kaç adımda bir ilerleme yazılsın")
    p.add_argument("--ornek-baslangic", default="Türkiye", help="değerlendirmede üretilecek örnek metnin başlangıcı")
    p.add_argument("--ornek-uzunluk", type=int, default=60, help="örnek metnin token sayısı (0: üretme)")
    p.add_argument("--cihaz", default="auto", help="auto, cpu, cuda veya mps")
    p.add_argument("--derle", action="store_true", help="torch.compile ile hızlandır (GPU'da önerilir)")
    p.add_argument("--devam", action="store_true", help="çıktı klasöründeki durum.pt'den eğitime devam et")
    p.add_argument("--baslangic-modeli", default=None,
                   help="sıfırdan değil, bu model.pt'nin ağırlıklarından başla (yeni veriyle devam eğitimi; "
                        "veri aynı tokenizer'la hazırlanmış olmalı)")
    p.add_argument("--tohum", type=int, default=1337, help="rastgelelik tohumu (tekrarlanabilirlik için)")
    return p


def cli(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    train(args)
