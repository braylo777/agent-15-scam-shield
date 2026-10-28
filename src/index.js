import { json, limited, body } from "./core.js";
import { classify, classifyEmbed } from "./app.js";
import page from "./page.html";
export default {
  async fetch(req, env) {
    const p = new URL(req.url).pathname;
    if (req.method === "POST" && (p === "/api/classify" || p === "/api/classify-embed")) {
      const { text } = await body(req);
      if (typeof text !== "string" || !text.trim() || text.length > 4000) return json({ error: "text required (<=4000 chars)" }, 400);
      if (p === "/api/classify") return json(classify(text));
      if (await limited(req, env)) return json({ error: "Rate limit: try again in a minute." }, 429);
      const r = await classifyEmbed(env, text, null).catch((e) => ({ error: "Model error: " + e.message }));
      return json(r, r.error ? 503 : 200);
    }
    if (p === "/api/health") return json({ ok: true });
    return new Response(page, { headers: { "content-type": "text/html;charset=utf-8" } });
  },
};
