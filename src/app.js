import { EMBED } from "./core.js";
import { score } from "./features.js";
import { MODEL_W } from "./model.js";
export const THRESHOLD = 0.5;
const SIGNALS = [["link", /https?:\/\/|www\./i], ["money", /[£$€]\s?\d|\bprize|cash\b/i], ["urgency", /\burgent|act now|expires?|suspended|verify\b/i], ["phone/shortcode", /\b\d{5,}\b/], ["credentials", /\bpassword|pin|ssn|login|account\b/i]];
export function classify(text) {
  const p = score(MODEL_W, text);
  return { label: p >= THRESHOLD ? "scam" : "legit", p_scam: +p.toFixed(4), signals: SIGNALS.filter(([, r]) => r.test(text)).map(([n]) => n), model: "hashed-ngram logistic regression (trained on box)" };
}
// Embedding head (bge-base + logistic). Needs Workers AI; head is trained once quota resets.
export async function classifyEmbed(env, text, head) {
  if (!head) return { error: "embedding head not yet trained (Workers AI quota reset pending)" };
  const r = await env.AI.run(EMBED, { text: [text] }); const v = r.data[0];
  let z = head.b; v.forEach((x, i) => (z += x * head.w[i])); const p = 1 / (1 + Math.exp(-z));
  return { label: p >= THRESHOLD ? "scam" : "legit", p_scam: +p.toFixed(4), model: "bge-base-en-v1.5 + logistic head" };
}
