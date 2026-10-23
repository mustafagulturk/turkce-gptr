// Sohbet arayüzü: mesajları ekrana basar, modeli çalıştıran Worker ile konuşur.
const $ = id => document.getElementById(id);
const chat = $("sohbet"), form = $("form"), input = $("girdi"), send = $("gonder");

let messages = [];      // [{role: "kullanici"|"asistan", content}]
let busy = false, current = null, maxTokens = 100;

const modelUrl = new URL("./model", import.meta.url).href;

// Hangi adımda takıldığı ekrandan anlaşılsın: "Sayfa açılıyor" (HTML) →
// "Model başlatılıyor" (bu dosya çalıştı) → "Model indiriliyor %0" (model kodu çalıştı).
$("yukleme-yazi").textContent = "Model başlatılıyor…";

// Önce modeli ayrı bir iş parçacığında (Worker) çalıştırmayı dene. Bazı tarayıcılar
// modül Worker'ını açamıyor; 8 saniye içinde ses gelmezse modeli sayfanın içinde
// çalıştır (cevap üretilirken sayfa biraz takılabilir ama çalışır).
let toModel, started = false;
try {
  const worker = new Worker(new URL("./worker.js?v=4", import.meta.url), { type: "module" });
  worker.onmessage = e => { started = true; onMessage(e.data); };
  worker.onerror = () => { if (!started) { worker.terminate(); runInPage(); } };
  toModel = msg => worker.postMessage(msg);
  toModel({ type: "load", url: modelUrl });
  setTimeout(() => { if (!started) { worker.terminate(); runInPage(); } }, 8000);
} catch {
  runInPage();
}

let inPage = false;
async function runInPage() {
  if (inPage) return;
  inPage = started = true;
  try {
    const { startInPage } = await import("./worker.js?v=4");
    const handle = startInPage(msg => setTimeout(() => onMessage(msg), 0));
    toModel = msg => handle(msg);
    toModel({ type: "load", url: modelUrl });
  } catch (e) {
    $("yukleme-yazi").textContent = `Model başlatılamadı: ${e.message || e}`;
  }
}

function onMessage(data) {
  if (data.type === "progress") {
    $("ilerleme").style.width = `${Math.round(data.value * 100)}%`;
    $("yukleme-yazi").textContent = `Model indiriliyor… %${Math.round(data.value * 100)}`;
  } else if (data.type === "error") {
    $("yukleme-yazi").textContent = `Model yüklenemedi: ${data.message}. Sayfayı yenilemeyi deneyin.`;
  } else if (data.type === "ready") {
    maxTokens = Math.min(100, Math.floor(data.config.block_size / 2));
    $("yukleme").hidden = true;
    $("oneriler").hidden = false;
    input.disabled = send.disabled = false;
    input.focus();
    const c = data.config;
    $("durum").textContent = `Model: ${c.n_layer} katman, genişlik ${c.n_embd}, bağlam ${c.block_size} token, sözlük ${c.vocab_size}`;
  } else if (data.type === "token") {
    current.content += data.text;
    current.el.textContent = current.content;
    chat.scrollTop = chat.scrollHeight;
  } else if (data.type === "done") {
    current.el.classList.remove("yaziyor");
    if (!current.content.trim()) current.el.textContent = "(boş cevap)";
    messages.push({ role: "asistan", content: current.content.trim() });
    $("durum").textContent = `${data.tokens} token, saniyede ${data.tokPerSec.toFixed(1)} token`;
    finish();
  }
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = `mesaj ${role}`;
  el.textContent = text;
  chat.append(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function ask(text) {
  text = text.trim();
  if (!text || busy) return;
  $("karsilama").hidden = true;
  bubble("kullanici", text);
  messages.push({ role: "kullanici", content: text });
  current = { el: bubble("asistan yaziyor", ""), content: "" };
  busy = true;
  send.textContent = "Durdur";
  input.value = "";
  toModel({
    type: "generate",
    messages,
    settings: { temperature: +$("sicaklik").value, topP: +$("topp").value, topK: 50, repetitionPenalty: 1.1, maxTokens },
  });
}

function finish() {
  busy = false;
  send.textContent = "Gönder";
  input.focus();
}

form.addEventListener("submit", e => {
  e.preventDefault();
  if (busy) toModel({ type: "stop" });
  else ask(input.value);
});
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});
$("oneriler").addEventListener("click", e => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
$("temizle").addEventListener("click", () => {
  if (busy) return;
  messages = [];
  chat.querySelectorAll(".mesaj").forEach(m => m.remove());
  $("karsilama").hidden = false;
});
for (const id of ["sicaklik", "topp"]) $(id).addEventListener("input", () => { $(`${id}-deger`).textContent = $(id).value; });
