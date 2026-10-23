// Model ayrı bir iş parçacığında (Web Worker) çalışır; böylece cevap üretilirken
// sayfa donmaz, yazı ekrana akarken kaydırma ve yazma devam eder.
// Worker açılamayan tarayıcılarda app.js bu dosyayı sayfanın içinde de çalıştırabilir;
// o yüzden mesajlaşma `post` fonksiyonuyla soyutlandı.
import { loadModel, sample } from "./motor.js?v=6";

let tokenizer = null, model = null, stopped = false;
let post = msg => self.postMessage(msg);

export function startInPage(send) {
  post = send;
  return data => handle({ data });
}

const inWorker = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;
if (inWorker) self.onmessage = e => handle(e);

async function handle({ data }) {
  if (data.type === "load") {
    try {
      ({ tokenizer, model } = await loadModel(data.url, p => post({ type: "progress", value: p })));
      post({ type: "ready", config: model.cfg });
    } catch (e) {
      // Hata sessizce kaybolmasın; sayfa "indiriliyor"da takılı kalmasın.
      post({ type: "error", message: String(e.message || e) });
    }
  } else if (data.type === "stop") {
    stopped = true;
  } else if (data.type === "generate") {
    stopped = false;
    generate(data.messages, data.settings);
  }
}

async function generate(messages, settings) {
  const sp = tokenizer.special;
  const stopIds = new Set([sp["<|son|>"], sp["<|kullanici|>"]]);
  const limit = model.cfg.block_size;
  // Bağlam sınırı: cevaba yer kalsın diye en eski turları at (interactive.py'deki gibi).
  let history = messages.slice();
  let ids = tokenizer.encodeChat(history);
  while (ids.length > limit - settings.maxTokens && history.length > 1) {
    history = history.slice(2);
    ids = tokenizer.encodeChat(history);
  }
  ids = ids.slice(-(limit - 1));

  model.reset();
  let logits;
  for (const t of ids) logits = model.forward(t); // prompt'u cache'e doldur

  const decoder = new TextDecoder("utf-8");
  const produced = [];
  const t0 = performance.now();
  for (let i = 0; i < settings.maxTokens && model.pos < limit && !stopped; i++) {
    const next = sample(logits, produced, settings);
    if (stopIds.has(next)) break;
    produced.push(next);
    // stream:true -> yarım kalan Türkçe harfi (ör. 'ş'nin ilk byte'ı) bekletir
    const text = decoder.decode(tokenizer.vocab[next], { stream: true });
    if (text) post({ type: "token", text });
    if (model.pos >= limit) break;
    logits = model.forward(next);
    // Kısa bir mola: bu sırada gelen "durdur" mesajı işlenebilsin.
    await new Promise(r => setTimeout(r, 0));
  }
  const tail = decoder.decode();
  if (tail) post({ type: "token", text: tail });
  const secs = (performance.now() - t0) / 1000;
  post({ type: "done", tokens: produced.length, tokPerSec: produced.length / Math.max(secs, 1e-3) });
}
