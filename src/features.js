// Model-free features: hashed word unigrams/bigrams + char 3-grams + a few scam signals. Pure JS, runs in the Worker.
export const DIM = 8192;
const h = (s) => { let x = 2166136261; for (let i = 0; i < s.length; i++) x = Math.imul(x ^ s.charCodeAt(i), 16777619); return (x >>> 0) % DIM; };
export function featurize(text) {
  const t = String(text || "").toLowerCase();
  const w = t.match(/[a-z0-9£$€]+/g) || [];
  const f = new Map(), add = (k, v = 1) => f.set(k, (f.get(k) || 0) + v);
  w.forEach((x, i) => { add(h("w:" + x)); if (i) add(h("b:" + w[i - 1] + "_" + x)); });
  const c = " " + t.replace(/\s+/g, " ") + " ";
  for (let i = 0; i + 3 <= c.length; i++) add(h("c:" + c.slice(i, i + 3)), 0.3);
  if (/https?:\/\/|www\.|\.(com|net|ly|co)\b/.test(t)) add(h("sig:url"), 2);
  if (/\b\d{5,}\b/.test(t)) add(h("sig:longnum"), 2);
  if (/[£$€]\s?\d/.test(t)) add(h("sig:money"), 2);
  if (/\b(urgent|verify|suspended|claim|winner|won|prize|act now|gift card|password|account)\b/.test(t)) add(h("sig:urgency"), 2);
  let n = 0; for (const v of f.values()) n += v * v; n = Math.sqrt(n) || 1;
  return [...f].map(([k, v]) => [k, v / n]);
}
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export function score(model, text) { let z = model.b; for (const [k, v] of featurize(text)) z += (model.w[k] || 0) * v; return sigmoid(z); }
