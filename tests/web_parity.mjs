// tests/test_web.py tarafından çağrılır: JS tokenizer ve modelinin çıktısını JSON olarak basar.
import { readFileSync } from "node:fs";
import { GPT, Tokenizer } from "../web/gpt.js";

const [dir, inputPath] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(`${dir}/model.json`, "utf-8"));
const bin = readFileSync(`${dir}/model.bin`);
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const input = JSON.parse(readFileSync(inputPath, "utf-8"));

const tok = new Tokenizer(meta.tokenizer);
const model = new GPT(meta, buffer);
const encoded = input.texts.map(t => tok.encode(t));
const chat = tok.encodeChat(input.chat);
const logits = input.tokens.map(t => Array.from(model.forward(t)));
console.log(JSON.stringify({ encoded, chat, logits }));
