// Çınar'ın tarayıcıda çalışan hali.
//
// PyTorch'taki turkce_gpt/model.py ve turkce_gpt/tokenizer.py'nin JavaScript
// karşılığı. Hiçbir kütüphane kullanmaz: tokenizer, matris-vektör çarpımı,
// RMSNorm, RoPE, dikkat ve SwiGLU burada elle yazıldı. Model dosyası
// `python -m turkce_gpt web-aktar` ile üretilir (model.json + model.bin).
//
// Tarayıcıda tek kullanıcı ve tek dizi olduğu için her şey "tek token" yolunda
// hesaplanır: her adımda bir token modele girer, önceki tokenlerin K/V'leri
// KV cache'te saklanır. Böylece her yeni token için iş miktarı sabit kalır.

// ------------------------------------------------------------- Tokenizer
// Python'daki PRETOKEN_PATTERN'in birebir karşılığı. Python'da [^\W\d_] "harf"
// demektir; JS'te bunu Unicode sınıflarıyla yazıyoruz (\p{L} harf, \p{Nd} rakam).
const PRETOKEN =
  /( ?[\p{L}\p{Nl}\p{No}]+| ?\p{Nd}{1,3}| ?[^\s\p{L}\p{Nl}\p{No}\p{Nd}_]+|\s+(?!\S)|\s+)/gu;

export class Tokenizer {
  constructor(meta) {
    // vocab[id] = o tokenin byte dizisi; ranks: "a,b" -> yeni token numarası
    this.vocab = [];
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    this.ranks = new Map();
    meta.merges.forEach(([a, b], i) => {
      const id = 256 + i;
      this.ranks.set(a * 65536 + b, id);
      const merged = new Uint8Array(this.vocab[a].length + this.vocab[b].length);
      merged.set(this.vocab[a]);
      merged.set(this.vocab[b], this.vocab[a].length);
      this.vocab.push(merged);
    });
    this.special = meta.special_tokens; // {"<|son|>": 8189, ...}
    for (const [tok, id] of Object.entries(this.special)) this.vocab[id] = new TextEncoder().encode(tok);
    this.cache = new Map();
    this.encoder = new TextEncoder();
  }

  encodeChunk(chunk) {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    let ids = Array.from(this.encoder.encode(chunk));
    // Her turda, uygulanabilecek EN ÖNCE öğrenilmiş birleştirmeyi uygula.
    while (ids.length >= 2) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = ids[best], b = ids[best + 1], out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(bestRank); i++; }
        else out.push(ids[i]);
      }
      ids = out;
    }
    this.cache.set(chunk, ids);
    return ids;
  }

  // Kullanıcı metni: özel tokenler (<|asistan|> gibi) TANINMAZ, düz metin sayılır.
  encode(text) {
    const ids = [];
    for (const m of text.normalize("NFC").matchAll(PRETOKEN)) ids.push(...this.encodeChunk(m[0]));
    return ids;
  }

  // Sohbet şablonu: <|kullanici|>soru<|asistan|>cevap<|son|> ...  (chat.py ile aynı)
  encodeChat(messages) {
    const ids = [];
    for (const m of messages) {
      ids.push(this.special[m.role === "kullanici" ? "<|kullanici|>" : "<|asistan|>"]);
      ids.push(...this.encode(m.content));
      if (m.role === "asistan") ids.push(this.special["<|son|>"]);
    }
    ids.push(this.special["<|asistan|>"]);
    return ids;
  }
}

// ------------------------------------------------------------- Model
export class GPT {
  constructor(meta, buffer) {
    const c = meta.config;
    this.cfg = c;
    this.hidden = meta.mlp_hidden;
    this.headDim = c.n_embd / c.n_head;
    const view = (info, Type) => new Type(buffer, info.offset, info.shape.reduce((a, b) => a * b, 1));
    // int8 matris: q (Int8Array) + satır ölçekleri (Float32Array)
    const mat = name => {
      const t = meta.tensors[name];
      return { q: view(t.q, Int8Array), scale: view(t.scale, Float32Array), rows: t.q.shape[0], cols: t.q.shape[1] };
    };
    const vec = name => view(meta.tensors[name].f32, Float32Array);
    this.emb = mat("tok_emb.weight");
    this.layers = [];
    for (let i = 0; i < c.n_layer; i++) {
      const p = `blocks.${i}.`;
      this.layers.push({
        attnNorm: vec(p + "attn_norm.weight"), qkv: mat(p + "attn.qkv.weight"), proj: mat(p + "attn.proj.weight"),
        mlpNorm: vec(p + "mlp_norm.weight"), gate: mat(p + "mlp.gate.weight"), up: mat(p + "mlp.up.weight"),
        down: mat(p + "mlp.down.weight"),
      });
    }
    this.norm = vec("norm.weight");

    // RoPE tabloları (model.py'deki precompute_rope ile aynı formül)
    const half = this.headDim / 2;
    this.cos = new Float32Array(c.block_size * this.headDim);
    this.sin = new Float32Array(c.block_size * this.headDim);
    for (let pos = 0; pos < c.block_size; pos++) {
      for (let i = 0; i < half; i++) {
        const angle = pos / Math.pow(c.rope_theta, (2 * i) / this.headDim);
        for (const j of [i, i + half]) {
          this.cos[pos * this.headDim + j] = Math.cos(angle);
          this.sin[pos * this.headDim + j] = Math.sin(angle);
        }
      }
    }
    this.reset();
  }

  // KV cache'i boşalt (yeni bir sohbet ya da bağlam taşınca)
  reset() {
    const size = this.cfg.block_size * this.cfg.n_embd;
    this.kCache = this.layers.map(() => new Float32Array(size));
    this.vCache = this.layers.map(() => new Float32Array(size));
    this.pos = 0;
  }

  // y = W x  (W int8 satırlar x ölçek). Modelin zamanının neredeyse tamamı burada geçer.
  matvec(m, x, out) {
    const { q, scale, rows, cols } = m;
    for (let r = 0, o = 0; r < rows; r++, o += cols) {
      let s = 0;
      for (let j = 0; j < cols; j++) s += q[o + j] * x[j];
      out[r] = s * scale[r];
    }
    return out;
  }

  rmsnorm(x, w) {
    let ss = 0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const inv = 1 / Math.sqrt(ss / x.length + 1e-6);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
    return out;
  }

  // Vektörün her başlığını konumuna göre döndür: x*cos + rotate_half(x)*sin
  rope(v, pos) {
    const hd = this.headDim, half = hd / 2, base = pos * hd;
    for (let h = 0; h < this.cfg.n_head; h++) {
      const o = h * hd;
      for (let i = 0; i < half; i++) {
        const a = v[o + i], b = v[o + i + half];
        v[o + i] = a * this.cos[base + i] - b * this.sin[base + i];
        v[o + i + half] = b * this.cos[base + i + half] + a * this.sin[base + i + half];
      }
    }
  }

  // Bir token işler, bir sonraki token için logit'leri döndürür.
  forward(token) {
    const { n_embd: C, n_head: H } = this.cfg, hd = this.headDim, pos = this.pos;
    if (pos >= this.cfg.block_size) throw new Error("bağlam dolu");
    // Embedding satırını int8'den geri çöz
    const x = new Float32Array(C);
    const er = token * C, es = this.emb.scale[token];
    for (let i = 0; i < C; i++) x[i] = this.emb.q[er + i] * es;

    const qkv = new Float32Array(3 * C), att = new Float32Array(C), tmp = new Float32Array(C);
    const g = new Float32Array(this.hidden), u = new Float32Array(this.hidden);
    const scores = new Float32Array(pos + 1);
    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l];
      // --- Dikkat
      this.matvec(L.qkv, this.rmsnorm(x, L.attnNorm), qkv);
      const q = qkv.subarray(0, C), k = qkv.subarray(C, 2 * C), v = qkv.subarray(2 * C);
      this.rope(q, pos);
      this.rope(k, pos);
      this.kCache[l].set(k, pos * C);
      this.vCache[l].set(v, pos * C);
      const K = this.kCache[l], V = this.vCache[l], sqrt = Math.sqrt(hd);
      for (let h = 0; h < H; h++) {
        const o = h * hd;
        // Sadece 0..pos (geçmiş + kendisi): causal maske kendiliğinden sağlanır.
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += q[o + i] * K[t * C + o + i];
          scores[t] = s / sqrt;
          if (scores[t] > max) max = scores[t];
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
        for (let i = 0; i < hd; i++) {
          let s = 0;
          for (let t = 0; t <= pos; t++) s += scores[t] * V[t * C + o + i];
          att[o + i] = s / sum;
        }
      }
      this.matvec(L.proj, att, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      // --- SwiGLU: down(silu(gate(x)) * up(x))
      const xn = this.rmsnorm(x, L.mlpNorm);
      this.matvec(L.gate, xn, g);
      this.matvec(L.up, xn, u);
      for (let i = 0; i < this.hidden; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      this.matvec(L.down, g, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    this.pos++;
    // Çıkış: embedding matrisiyle çarp (weight tying)
    return this.matvec(this.emb, this.rmsnorm(x, this.norm), new Float32Array(this.cfg.vocab_size));
  }
}

// ------------------------------------------------------------- Örnekleme
// model.py'deki _sample ile aynı adımlar: tekrar cezası, sıcaklık, top-k, top-p.
export function sample(logits, history, { temperature = 0.7, topK = 50, topP = 0.9, repetitionPenalty = 1.1 } = {},
                       rand = Math.random) {
  const l = Float32Array.from(logits);
  if (repetitionPenalty !== 1) for (const id of new Set(history)) l[id] = l[id] > 0 ? l[id] / repetitionPenalty : l[id] * repetitionPenalty;
  if (temperature <= 0) return l.indexOf(Math.max(...l));
  let idx = Array.from(l.keys()).sort((a, b) => l[b] - l[a]).slice(0, topK > 0 ? topK : l.length);
  const max = l[idx[0]];
  let probs = idx.map(i => Math.exp((l[i] - max) / temperature));
  let sum = probs.reduce((a, b) => a + b, 0);
  probs = probs.map(p => p / sum);
  if (topP < 1) {
    let cum = 0, keep = 0;
    while (keep < probs.length && cum <= topP) cum += probs[keep++];
    idx = idx.slice(0, keep); probs = probs.slice(0, keep);
    sum = probs.reduce((a, b) => a + b, 0);
    probs = probs.map(p => p / sum);
  }
  let r = rand();
  for (let i = 0; i < idx.length; i++) { r -= probs[i]; if (r <= 0) return idx[i]; }
  return idx[idx.length - 1];
}

export async function loadModel(baseUrl, onProgress = () => {}) {
  // cache: "no-cache" -> tarayıcı önbellekteki kopyayı kullanmadan önce sunucuya
  // "değişti mi?" diye sorar. Model güncellenince eski model.json ile yeni
  // model.bin'in karışmasını önler (değişmediyse dosya tekrar indirilmez).
  const get = async name => {
    const r = await fetch(`${baseUrl}/${name}`, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${name} indirilemedi (HTTP ${r.status})`);
    return r;
  };
  const meta = await (await get("model.json")).json();
  const res = await get("model.bin");
  const reader = res.body.getReader();
  const buf = new Uint8Array(meta.bin_bytes);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > buf.length) throw new Error("model.bin, model.json ile uyuşmuyor");
    buf.set(value, got);
    got += value.length;
    // Sunucu dosyayı sıkıştırıp gönderebildiği için content-length'e değil,
    // model.json'daki gerçek boyuta göre ilerleme hesaplanır.
    onProgress(got / buf.length);
  }
  if (got !== buf.length) throw new Error("model.bin eksik indi");
  return { tokenizer: new Tokenizer(meta.tokenizer), model: new GPT(meta, buf.buffer) };
}
