"""
Modeli tarayıcıda çalışacak biçime çevirir:  python -m turkce_gpt web-aktar

Tarayıcıda PyTorch yok. Bu yüzden ağırlıkları düz bir ikili dosyaya (model.bin)
yazıyoruz; web/gpt.js bu dosyayı okuyup ileri geçişi JavaScript'te yapar.
Yanına model.json yazılır: model ayarları, tokenizer birleştirmeleri ve
model.bin içinde hangi ağırlığın nerede durduğu (indeks).

Boyutu küçültmek için int8 nicemleme (quantization):
    float32 bir ağırlık 4 byte tutar. Her matris SATIRINI kendi en büyük mutlak
    değerine göre -127..127 aralığına ölçekleyip 1 byte'lık tam sayı olarak
    saklarız; satır başına bir de float32 ölçek (scale) tutarız:
        w ≈ q * scale,   scale = max(|satır|) / 127
    Dosya ~4 kat küçülür (kucuk model: ~136 MB -> ~35 MB), kalite kaybı çok azdır.
    Normalizasyon ağırlıkları küçük olduğu için float32 kalır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .tokenizer import SPECIAL_TOKENS


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[satır, sütun] float32 matrisi satır bazında int8'e çevirir: (q, scale)."""
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0  # tamamen sıfır satırda bölme hatası olmasın
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def export(model_path: str | Path, out_dir: str | Path) -> dict:
    model, tokenizer, _ = load_checkpoint(model_path, device="cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, dict] = {}
    chunks: list[bytes] = []
    offset = 0

    def add(name: str, data: np.ndarray) -> dict:
        nonlocal offset
        raw = data.tobytes()
        # Her parçayı 4 byte'a hizalıyoruz ki JS'te Float32Array doğrudan açılabilsin.
        pad = (-len(raw)) % 4
        info = {"offset": offset, "dtype": str(data.dtype), "shape": list(data.shape)}
        chunks.append(raw + b"\0" * pad)
        offset += len(raw) + pad
        return info

    for name, param in model.state_dict().items():
        w = param.detach().float().numpy()
        if name == "lm_head.weight":
            continue  # tok_emb ile aynı (weight tying); tekrar yazmaya gerek yok
        if w.ndim == 2:
            q, scale = quantize_rows(w)
            tensors[name] = {"q": add(name, q), "scale": add(name + ".scale", scale)}
        else:
            tensors[name] = {"f32": add(name, w.astype(np.float32))}

    (out_dir / "model.bin").write_bytes(b"".join(chunks))

    meta = {
        "tur": "turkce-gpt-web",
        "surum": 1,
        "config": model.config.to_dict(),
        "mlp_hidden": model.blocks[0].mlp.up.weight.shape[0],
        "tokenizer": {
            "merges": [[a, b] for (a, b) in tokenizer.merges],
            "special_tokens": {tok: tokenizer.special_id(tok) for tok in SPECIAL_TOKENS},
        },
        "tensors": tensors,
        "bin_bytes": offset,
    }
    (out_dir / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def cli(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="python -m turkce_gpt web-aktar",
        description="Modeli tarayıcıda çalışacak biçime (model.json + int8 model.bin) çevirir.",
    )
    p.add_argument("--model", required=True, help="model.pt yolu (sohbet için ince ayarlı model önerilir)")
    p.add_argument("--cikti", default="web/model", help="model.json ve model.bin'in yazılacağı klasör")
    args = p.parse_args(argv)
    meta = export(args.model, args.cikti)
    mb = meta["bin_bytes"] / 1e6
    print(f"Yazıldı: {args.cikti}/model.json ve model.bin ({mb:.1f} MB, int8)")


if __name__ == "__main__":
    import sys

    torch.set_grad_enabled(False)
    cli(sys.argv[1:])
