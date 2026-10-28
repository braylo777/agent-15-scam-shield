# agent-15 · Scam Shield

**Live demo:** https://agent-15-scam-shield.braylo777.workers.dev

## What
Paste a text, email or DM and get `scam` / `legit`, a calibrated probability, and the signals that fired (link, money, urgency, shortcode, credentials). Runs entirely inside a Cloudflare Worker.

## Why
Scam texts and phishing DMs keep getting better as AI writes them. A small, explainable, cheap classifier at the edge is the first filter: it answers in milliseconds with no per-request LLM cost.

## How it was built
- **Data:** UCI SMS Spam Collection (5,574 labeled messages, 747 spam), in `evals/data/`.
- **Model:** hashed word unigrams and bigrams, char 3-grams and 4 scam-signal features (8,192 dims, L2-normalised), plus logistic regression trained with SGD in plain JS (`evals/train.mjs`, 30 epochs, L2 1e-4). Weights ship in `src/model.js`. No Python, no GPU, no LLM.
- **Split:** stratified 80/20, seed 42 (4,460 train / 1,114 test).
- **Embedding head (planned):** `/api/classify-embed` uses Workers AI `bge-base-en-v1.5` embeddings plus a logistic head. It returns 503 until the head is trained.

## Results (held-out test set, n=1,114, threshold 0.5). Real, from `evals/results.json`
| metric | value |
|---|---|
| precision | **1.000** (0 false positives) |
| recall | **0.919** (137/149 scams caught) |
| F1 | **0.958** |
| ROC-AUC | **0.998** |
| calibration ECE (10-bin) | **0.0096** |
| Brier score | **0.0089** |

The full reliability table is in `evals/results.json`.
**Embedding head (bge-base + LR): not yet run.** The Workers AI free daily quota was exhausted until 7 PM CT Oct 28, 2026.

Caveat: the data is 2000s-era UK SMS spam. Modern AI-voice and email phishing will look different, and this needs a newer eval set.

## How it ships
`npm test` (11 tests, no model calls), then `npm run train`, then `npm run build` (bundles to `dist/worker.js`), then deploy with `wrangler deploy` or the Cloudflare API upload. Free tier only, with a 5/min per-IP rate limit on the model endpoint. Rollback tag: `v0-rollback`.

## How to run

```bash
npm install   # dev deps only
npm test      # 15 offline tests, no Workers AI calls
npm run dev   # local wrangler dev
```

## Test results (2026-10-27)

| Suite | Tests | Status |
|---|---|---|
| Unit + edge cases (test/edge.test.js added) | 11→15 | all pass |
| Live demo https://agent-15-scam-shield.braylo777.workers.dev | HTTP 200 | verified 2026-10-27 |
