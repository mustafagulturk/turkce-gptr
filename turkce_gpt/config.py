"""
Model ayarları.

Bir GPT modelinin "boyutu" birkaç sayıyla belirlenir:
    vocab_size : tokenizer'daki token sayısı (modelin bildiği "kelime parçası" sayısı)
    block_size : modelin aynı anda görebildiği en fazla token sayısı (bağlam uzunluğu)
    n_layer    : üst üste konan transformer bloğu sayısı (derinlik)
    n_head     : dikkat (attention) başlığı sayısı; her başlık farklı bir ilişkiye bakar
    n_embd     : her tokeni temsil eden vektörün uzunluğu (genişlik)

Parametre sayısı kabaca 12 * n_layer * n_embd^2 kadardır. GPT-2 small:
12 katman, 768 genişlik -> ~124 milyon parametre.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    # RoPE (dönen konum kodlaması) için taban frekans. Llama'daki değerle aynı.
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd, n_head'e tam bölünmeli")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE için başlık boyutu (n_embd / n_head) çift olmalı")

    def to_dict(self) -> dict:
        return asdict(self)


# Hazır boyutlar. vocab_size, veri hazırlanırken tokenizer'dan alınır.
PRESETS: dict[str, dict] = {
    # Testler ve hızlı denemeler için (birkaç yüz bin parametre).
    "deneme": dict(block_size=64, n_layer=2, n_head=2, n_embd=64),
    # Dizüstü bilgisayarın işlemcisinde dakikalar içinde eğitilebilir (~4M).
    "mini": dict(block_size=128, n_layer=4, n_head=4, n_embd=192),
    # Colab'ın ücretsiz T4 GPU'sunda ~1 saatte anlamlı sonuç verir (~25M).
    "kucuk": dict(block_size=512, n_layer=8, n_head=8, n_embd=512),
    # Daha güçlü GPU (A100 vb.) için (~85M).
    "orta": dict(block_size=1024, n_layer=12, n_head=12, n_embd=768),
}


def make_config(preset: str, vocab_size: int, **overrides) -> GPTConfig:
    if preset not in PRESETS:
        raise ValueError(f"Bilinmeyen boyut '{preset}'. Seçenekler: {', '.join(PRESETS)}")
    values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    return GPTConfig(vocab_size=vocab_size, **values)
