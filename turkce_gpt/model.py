"""
GPT modeli (Llama tarzı), sıfırdan.

Bir dil modelinin tek işi şudur: önceki tokenlere bakıp SIRADAKİ tokenin
olasılık dağılımını tahmin etmek.
    "Bugün hava çok" -> {" güzel": 0.31, " sıcak": 0.22, " soğuk": 0.18, ...}
Metin üretmek de bu tahmini tekrar tekrar yapıp her seferinde bir token
seçmekten ibarettir.

Modelin akışı (bir token dizisi için):
    1. Token embedding : her token numarası öğrenilebilir bir vektöre çevrilir.
    2. N tane transformer bloğu, her biri:
         x = x + Attention(RMSNorm(x))   # tokenler birbirinden bilgi toplar
         x = x + MLP(RMSNorm(x))         # her token kendi bilgisini işler
    3. Son RMSNorm + lm_head : her konumdaki vektör, sözlükteki her token için
       bir puana (logit) çevrilir. Softmax bu puanları olasılığa çevirir.

GPT-2'ye göre Llama'dan aldığımız iyileştirmeler:
    - LayerNorm yerine RMSNorm (daha basit, aynı başarı)
    - Öğrenilen konum embedding'i yerine RoPE (dönen konum kodlaması)
    - GELU'lu MLP yerine SwiGLU (aynı parametreyle biraz daha iyi)
    - Doğrusal katmanlarda bias yok

"x = x + ..." şeklindeki "artık bağlantı" (residual connection) çok önemlidir:
gradyan bu kısa yoldan ilk katmanlara kadar zayıflamadan akar, böylece derin
ağlar eğitilebilir. "Pre-norm" (normu alt katmandan ÖNCE uygulamak) da
eğitimi kararlı yapar.
"""

from __future__ import annotations

import inspect
import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


# ---------------------------------------------------------------- KV cache
class KVCache:
    """Üretim sırasında her katmanın geçmiş key/value tensörlerini saklar.

    Neden hız kazandırır?
        Metin üretirken her adımda yalnızca BİR yeni token ekleriz. Cache
        olmasaydı her adımda tüm diziyi baştan modele verip geçmiş tokenlerin
        key/value'larını tekrar tekrar hesaplardık (adım başına O(T) iş, toplam
        O(T^2)). Oysa geçmiş tokenlerin K ve V'si değişmez (causal mask
        sayesinde geçmiş tokenler gelecekten etkilenmez). Onları bir kere
        hesaplayıp saklarsak her adımda sadece yeni tokenin Q, K, V'sini
        hesaplarız; yeni token, saklanan tüm K/V'lere bakar.

    Yapı: katman başına bir (k, v) ikilisi, her biri [B, n_head, T, head_dim].
    """

    def __init__(self, n_layer: int) -> None:
        self.layers: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layer

    def __len__(self) -> int:
        """Cache'te şu an kaç token saklandığını döndürür (ilk katmana bakarak)."""
        first = self.layers[0]
        return 0 if first is None else first[0].size(2)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Yeni k/v'yi o katmanın geçmişine ekler, birleşik (tam) k/v'yi döndürür."""
        past = self.layers[layer_idx]
        if past is not None:
            # Zaman ekseni (dim=2) boyunca geçmişin sonuna ekle.
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        self.layers[layer_idx] = (k, v)
        return k, v


# ---------------------------------------------------------------- RMSNorm
class RMSNorm(nn.Module):
    """Root Mean Square normalizasyonu.

    Her token vektörünü "ortalama büyüklüğü" 1 olacak şekilde ölçekler:
        y = x / sqrt(mean(x^2) + eps) * weight
    LayerNorm'dan farkı: ortalamayı çıkarmaz, sadece ölçeği düzeltir. Daha az
    hesap, pratikte aynı başarı. Amaç, katmanlar boyunca değerlerin patlamasını
    veya sönmesini engelleyip eğitimi kararlı tutmak. 'weight' ise modelin her
    boyutu isterse yeniden büyütüp küçültebilmesi için öğrenilen bir çarpandır.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sayısal kararlılık için hesabı float32'de yapıp eski tipe geri dönüyoruz
        # (fp16/bf16 eğitimde kareler taşabilir veya hassasiyet kaybolabilir).
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


# ---------------------------------------------------------------- RoPE
def precompute_rope(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE için her konum ve frekans çiftine ait cos/sin tablolarını hazırlar.

    RoPE'nin fikri:
        Attention, q ve k vektörlerinin iç çarpımına (benzerliğine) bakar. Biz
        q'yu ve k'yı bulundukları konuma göre DÖNDÜRÜRSEK, iki vektörün iç
        çarpımı yalnızca aradaki AÇI FARKINA, yani iki tokenin GÖRELİ uzaklığına
        bağlı olur:  <rot(q, m), rot(k, n)> = f(q, k, m - n).
        Böylece model "bu kelime 3 token önce geçti" gibi göreli bilgiyi
        doğrudan öğrenir; ayrıca ek bir konum parametresi gerekmez.

    Vektörün boyutları çiftlere ayrılır; her çift 2 boyutlu bir düzlemde
    farklı hızda döner. i'inci çiftin frekansı:  theta^(-2i / head_dim)
    İlk çiftler hızlı döner (yakın konumları ayırt eder), son çiftler yavaş
    döner (uzak konumları ayırt eder) — saatin saniye/dakika/saat kolları gibi.

    Döndürür: cos, sin — her biri [max_len, head_dim]
    """
    # Her çift için dönme hızı: [head_dim / 2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    # Konum m'deki açı = m * frekans  -> [max_len, head_dim / 2]
    angles = torch.outer(positions, inv_freq)
    # "rotate_half" yerleşimine uysun diye açıları iki kez yan yana koyuyoruz:
    # boyut i ile boyut i + head_dim/2 aynı düzlemin iki eksenidir.
    angles = torch.cat([angles, angles], dim=-1)  # [max_len, head_dim]
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Vektörü ikiye böler: (x1, x2) -> (-x2, x1). 2B'de 90 derece döndürmedir."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x'i konumuna göre döndürür. x: [B, n_head, T, head_dim], cos/sin: [T, head_dim].

    2B dönme formülü:  (a, b) -> (a*cos - b*sin, a*sin + b*cos)
    Bu, tam olarak  x*cos + rotate_half(x)*sin  ifadesine denktir.
    """
    return (x * cos + rotate_half(x) * sin).type_as(x)


# ---------------------------------------------------------------- Attention
class CausalSelfAttention(nn.Module):
    """Çok başlıklı, nedensel (causal) öz-dikkat.

    Sezgi (Q / K / V):
        Her token üç vektör üretir:
          Query (Q): "Ben neye ihtiyaç duyuyorum?"  (soru)
          Key   (K): "Bende ne var?"                (etiket)
          Value (V): "Bana bakan olursa ne vereyim?" (içerik)
        Bir tokenin Q'su ile diğer tokenlerin K'ları karşılaştırılır (iç çarpım);
        benzerlik yüksekse o tokenin V'sinden daha çok bilgi alınır. Sonuç,
        V'lerin ağırlıklı ortalamasıdır:
            Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

    Neden sqrt(d) ile bölüyoruz?
        d boyutlu, elemanları ~N(0,1) olan iki vektörün iç çarpımının varyansı
        d'dir; yani d büyüdükçe puanlar büyür. Çok büyük puanlar softmax'ı
        "tek bir tokene %100" haline doyurur ve gradyanlar neredeyse sıfıra
        iner. sqrt(d)'ye bölmek varyansı tekrar ~1'e çeker.

    Neden maske (mask)?
        Model eğitimde tüm diziyi aynı anda görür ama her konumda SIRADAKİ
        tokeni tahmin etmeli. Gelecekteki tokenlere bakabilseydi cevabı
        kopyalardı (kopya çekme!). Causal mask, her tokenin yalnızca kendisine
        ve öncesine bakmasını sağlar: gelecek konumların puanları -sonsuz
        yapılır, softmax sonrası ağırlıkları 0 olur.

    Neden çok başlık (multi-head)?
        Her başlık farklı bir ilişki türünü öğrenebilir (ör. biri özne-yüklem
        uyumuna, biri bir önceki kelimeye bakar). Vektör n_head parçaya bölünür,
        her parça kendi attention'ını hesaplar, sonra sonuçlar birleştirilir.
    """

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        # Q, K, V'yi tek bir büyük matris çarpımıyla birlikte hesaplıyoruz (daha hızlı).
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Başlıkların çıktısını birleştirip tekrar n_embd boyutuna karıştıran katman.
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # True: PyTorch'un hızlı (FlashAttention vb.) yolu; False: elle yazılmış, öğretici yol.
        self.use_sdpa = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]: her başlık ayrı bir "kanal" olur.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Konum bilgisini Q ve K'ya döndürme ile ekle (V'ye eklenmez; konum
        # yalnızca "kime bakılacağını" etkilemeli, taşınan içeriği değil).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache: geçmiş tokenlerin k/v'si saklıysa yenilerini ekle.
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
        T_total = k.size(2)       # geçmiş + yeni token sayısı
        past_len = T_total - T    # cache'ten gelen (geçmiş) token sayısı

        if self.use_sdpa:
            dropout_p = self.dropout if self.training else 0.0
            if past_len == 0:
                # Standart durum: kare causal maske, SDPA kendisi uygular.
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            elif T == 1:
                # Tek yeni token: geçmişin tamamını görebilir, maskeye gerek yok.
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                # Birden çok yeni token + geçmiş: is_causal sol-üst hizalı olduğu
                # için yanlış olurdu; maskeyi kendimiz kuruyoruz (True = bakabilir).
                mask = self._causal_mask(T, T_total, past_len, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            # --- Elle yazılmış yol: formülün birebir karşılığı ---
            # 1) Benzerlik puanları: her sorgu her anahtarla. [B, nh, T, T_total]
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 2) Gelecek konumları -sonsuz yap (softmax'ta 0 olsunlar).
            mask = self._causal_mask(T, T_total, past_len, x.device)
            att = att.masked_fill(~mask, float("-inf"))
            # 3) Satır bazında softmax: her token için ağırlıkların toplamı 1.
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 4) Value'ların ağırlıklı ortalaması. [B, nh, T, head_dim]
            y = att @ v

        # Başlıkları yeniden yan yana koy: [B, nh, T, hd] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))

    @staticmethod
    def _causal_mask(T: int, T_total: int, past_len: int, device: torch.device) -> torch.Tensor:
        """[T, T_total] boyutlu bool maske. Yeni i'inci token (mutlak konumu
        past_len + i) 0..past_len+i arasındaki anahtarlara bakabilir."""
        ones = torch.ones(T, T_total, dtype=torch.bool, device=device)
        return torch.tril(ones, diagonal=past_len)


# ---------------------------------------------------------------- MLP
class SwiGLU(nn.Module):
    """SwiGLU ileri beslemeli ağ (feed-forward).

    Attention tokenler ARASINDA bilgi taşır; MLP ise her tokenin vektörünü
    TEK BAŞINA işler ("düşünme" kısmı). Modelin bilgisinin büyük kısmı
    buradaki ağırlıklarda saklanır.

        SwiGLU(x) = down( silu(gate(x)) * up(x) )

    'gate' kolu bir "kapı" gibi çalışır: silu(gate(x)) her boyut için ne kadar
    bilginin geçeceğine karar verir, 'up' kolu ise geçecek içeriği üretir.
    Klasik MLP'de 2 matris (4*n_embd genişlik) varken burada 3 matris var; aynı
    parametre sayısını korumak için gizli boyutu 4 yerine ~8/3 * n_embd
    seçiyoruz (Llama'daki gibi) ve GPU verimliliği için 64'ün katına yuvarlıyoruz.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)  # yukarı, 64'ün katına yuvarla
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------- Blok
class Block(nn.Module):
    """Bir transformer bloğu: önce iletişim (attention), sonra işleme (MLP)."""

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None) -> torch.Tensor:
        # Pre-norm + artık bağlantı: x'in kendisi hiç bozulmadan yukarı akar,
        # alt katmanlar ona sadece "düzeltme" ekler.
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------- GPT
class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: girişteki embedding matrisi ile çıkıştaki lm_head aynı
        # matristir. Mantığı: "bir tokeni vektöre çevirmek" ile "bir vektörün
        # hangi tokene benzediğini ölçmek" aynı bilgiyi kullanır. Küçük
        # modellerde parametrelerin büyük kısmını kurtarır ve genelde daha iyi
        # sonuç verir.
        self.lm_head.weight = self.tok_emb.weight

        # RoPE cos/sin tablolarını bir kere hesaplayıp buffer olarak saklıyoruz.
        # persistent=False: state_dict'e yazılmazlar (her zaman yeniden
        # hesaplanabilirler), ama .to(device) ile modelle birlikte taşınırlar.
        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # GPT-2 tarzı başlangıç ağırlıkları.
        self.apply(self._init_weights)
        # Artık yola eklenen çıkış projeksiyonlarını daha küçük başlatıyoruz:
        # her blok artık akışa 2 katkı ekler (attention + MLP); toplam
        # 2*n_layer katkının varyansı büyümesin diye std'yi sqrt(2*n_layer)'e
        # bölüyoruz. Böylece başlangıçta derin modelde de aktivasyonlar patlamaz.
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Küçük rastgele değerler (std 0.02): başlangıçta tüm tokenlere yaklaşık
        # eşit olasılık verilir, yani loss ~ ln(vocab_size) olur.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Toplam (benzersiz) parametre sayısı. Paylaşılan embedding/lm_head bir kez sayılır."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_cache: KVCache | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: [B, T] token numaraları. targets: [B, T] (her konum için doğru
        sıradaki token; -100 olan konumlar loss'a katılmaz).

        Döndürür: logits [B, T, vocab_size], loss (targets yoksa None).
        """
        B, T = idx.shape
        # Cache varsa yeni tokenlerin konumu, cache'teki token sayısından başlar.
        start = len(kv_cache) if kv_cache is not None else 0
        if start + T > self.config.block_size:
            raise ValueError(
                f"Dizi uzunluğu ({start + T}) bağlam uzunluğunu ({self.config.block_size}) aşıyor"
            )
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        x = self.dropout(self.tok_emb(idx))  # [B, T, n_embd]
        for block in self.blocks:
            x = block(x, cos, sin, kv_cache)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Cross-entropy = doğru tokene verilen olasılığın -log'u (ortalaması).
            # Rastgele tahminde her token 1/V olasılık alır -> loss = ln(V).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def configure_optimizer(self, weight_decay: float, lr: float,
                            betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
        """AdamW optimizer'ı iki parametre grubuyla kurar.

        Weight decay ağırlıkları her adımda biraz sıfıra çeker (aşırı öğrenmeye
        karşı düzenlileştirme). Bunu yalnızca 2+ boyutlu matrislere (Linear,
        Embedding) uygularız; RMSNorm ağırlıkları ve bias gibi 1 boyutlu
        parametreleri küçültmek bir fayda sağlamaz, hatta zarar verebilir.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # CUDA'da "fused" AdamW daha hızlıdır (destekleniyorsa kullan).
        extra = {}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters and decay and decay[0].is_cuda:
            extra["fused"] = True
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    # ------------------------------------------------------------ üretim
    @staticmethod
    def _sample(logits: torch.Tensor, history: list[int], temperature: float, top_k: int | None,
                top_p: float | None, repetition_penalty: float) -> int:
        """Son konumun logit'lerinden [vocab] bir sonraki tokeni seçer."""
        logits = logits.float().clone()

        # Tekrar cezası (CTRL makalesi): bağlamda geçmiş tokenlerin puanını düşür.
        # Pozitif puan bölünür, negatif puan çarpılır -> her iki durumda da azalır.
        if repetition_penalty != 1.0 and history:
            seen = torch.tensor(sorted(set(history)), device=logits.device)
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        # Sıcaklık 0: her zaman en olası token (greedy, deterministik).
        if temperature <= 0:
            return int(torch.argmax(logits).item())

        # Sıcaklık: puanları T'ye bölmek dağılımı T<1'de sivriltir (daha emin,
        # daha tekdüze), T>1'de düzleştirir (daha yaratıcı, daha hatalı).
        logits = logits / temperature

        # Top-k: yalnızca en olası k token arasından seç; uzun kuyruktaki saçma
        # tokenleri ele.
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[-1]
            logits[logits < kth] = float("-inf")

        # Top-p (nucleus): olasılıkları büyükten küçüğe topla; toplam p'yi
        # geçene kadarki en küçük kümeyi tut. Model eminse küme küçük, kararsızsa
        # büyük olur — top-k'nın sabit k'sından daha esnektir.
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # Kendisinden ÖNCEKİLERİN toplamı zaten p'yi geçmişse at (ilk token hep kalır).
            remove = (cum - probs) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                        top_k: int | None = 50, top_p: float | None = 0.95,
                        repetition_penalty: float = 1.0,
                        stop_ids: set[int] | None = None) -> Iterator[int]:
        """Başlangıç dizisinden [1, T] devam ederek her adımda bir token id'si üretir.

        KV cache kullanır: önce tüm başlangıç dizisi bir kere işlenir, sonra her
        adımda yalnızca yeni token modele verilir. stop_ids'ten bir token
        seçilirse onu yield ETMEDEN durur. Bağlam block_size'ı aşacaksa son
        block_size tokenle cache baştan kurulur (model en fazla block_size
        token görebilir).
        """
        block_size = self.config.block_size
        stop_ids = stop_ids or set()
        tokens: list[int] = idx[0].tolist()
        if not tokens:
            raise ValueError("generate_stream en az bir başlangıç tokeni ister")
        device = idx.device

        # Üretim sırasında dropout kapalı olmalı; işimiz bitince eski moda dön.
        was_training = self.training
        self.eval()
        try:
            def prefill(window: list[int]) -> tuple[KVCache, torch.Tensor]:
                cache = KVCache(self.config.n_layer)
                x = torch.tensor([window], dtype=torch.long, device=device)
                logits, _ = self(x, kv_cache=cache)
                return cache, logits[0, -1]

            cache, last_logits = prefill(tokens[-block_size:])
            for _ in range(max_new_tokens):
                window = tokens[-block_size:]
                next_id = self._sample(last_logits, window, temperature, top_k, top_p, repetition_penalty)
                if next_id in stop_ids:
                    return
                yield next_id
                tokens.append(next_id)

                if len(cache) + 1 > block_size:
                    # Cache dolu: kaydırmalı pencere ile baştan kur. (RoPE konumları
                    # 0'dan başlamalı, o yüzden eski cache'i kırpamayız.)
                    cache, last_logits = prefill(tokens[-block_size:])
                else:
                    x = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    logits, _ = self(x, kv_cache=cache)
                    last_logits = logits[0, -1]
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kw) -> torch.Tensor:
        """generate_stream'in topluca hali: [1, T] -> [1, T + n] (n <= max_new_tokens)."""
        new_ids = list(self.generate_stream(idx, max_new_tokens, **kw))
        if not new_ids:
            return idx
        extra = torch.tensor([new_ids], dtype=idx.dtype, device=idx.device)
        return torch.cat([idx, extra], dim=1)
