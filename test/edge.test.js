import { test } from "node:test";
import assert from "node:assert/strict";
import { classify } from "../src/app.js";
import worker from "../src/index.js";
const post = (b) => worker.fetch(new Request("http://x/api/classify", { method: "POST", body: b }), {});
test("edge: unicode/emoji text classifies with valid prob", () => { const r = classify("🎉🎉 héllo ñandú 你好"); assert.ok(r.p_scam >= 0 && r.p_scam <= 1); });
test("edge: very long text does not throw", () => { assert.doesNotThrow(() => classify("ok ".repeat(50000))); });
test("edge: classify is deterministic", () => { const t = "URGENT claim your reward now"; assert.equal(classify(t).p_scam, classify(t).p_scam); });
test("edge: malformed JSON body is a 4xx not 5xx", async () => { const s = (await post("{not json")).status; assert.ok(s >= 400 && s < 500, String(s)); });
