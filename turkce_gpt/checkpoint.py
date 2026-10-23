"""
Modeli kaydetme ve yükleme (checkpoint).

Bir modeli başka bir bilgisayarda (ör. Colab'da eğitip kendi dizüstünde)
kullanabilmek için üç şeye ihtiyaç var:
    1. Ağırlıklar (state_dict): eğitimde öğrenilen tüm sayılar.
    2. Ayarlar (config): katman sayısı, genişlik... Ağırlıkları yerleştireceğimiz
       boş modeli aynı şekilde kurabilmek için.
    3. Tokenizer: metni AYNI numaralara çevirmek için. Farklı tokenizer'la
       model anlamsız sonuç verir; çünkü 312 numaralı token artık başka bir
       parça demektir.

Hepsini tek bir .pt dosyasına koyuyoruz; böylece dosyayı kopyalamak yeterli.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import BPETokenizer


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """torch.compile ile derlenmiş modelin içindeki asıl modeli döndürür.

    Derlenmiş modelin state_dict anahtarları "_orig_mod." önekiyle başlar;
    kaydetmeden önce asıl modele ulaşıp temiz anahtarlar kullanırız.
    """
    return getattr(model, "_orig_mod", model)


def tokenizer_to_dict(tokenizer: BPETokenizer) -> dict:
    """Tokenizer'ı, tokenizer.json dosyasıyla birebir aynı içerikte bir sözlüğe çevirir."""
    # Tokenizer'ın kendi save() biçimini kullanıyoruz ki iki biçim hiç ayrışmasın.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        tokenizer.save(path)
        return json.loads(path.read_text(encoding="utf-8"))


def tokenizer_from_dict(data: dict) -> BPETokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return BPETokenizer.load(path)


def save_checkpoint(path: str | Path, model: torch.nn.Module, tokenizer: BPETokenizer, extra: dict) -> None:
    """Model + ayarlar + tokenizer + ek bilgileri (adım, val_loss...) tek dosyaya yazar."""
    model = _unwrap(model)
    checkpoint = {
        "model": model.state_dict(),
        "config": model.config.to_dict(),
        "tokenizer": tokenizer_to_dict(tokenizer),
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Önce geçici dosyaya yazıp sonra adını değiştiriyoruz: kayıt sırasında
    # program kesilirse eski (sağlam) dosya bozulmamış olur.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, BPETokenizer, dict]:
    """Kaydedilmiş modeli kurar ve (model, tokenizer, ek_bilgiler) döndürür.

    Model değerlendirme (eval) modunda döner: dropout kapalıdır.
    """
    # weights_only=True: dosyadan sadece tensör, sayı, metin, liste, sözlük
    # okunur. pickle ile keyfi Python kodu çalıştırılamaz; internetten
    # indirilen bir .pt dosyasını açarken bu önemli bir güvenlik önlemidir.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = GPTConfig(**checkpoint.pop("config"))
    model = GPT(config)
    model.load_state_dict(checkpoint.pop("model"))
    model.to(device)
    model.eval()
    tokenizer = tokenizer_from_dict(checkpoint.pop("tokenizer"))
    return model, tokenizer, checkpoint
