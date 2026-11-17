# agent-15 · Scam Shield

**Live demo:** https://agent-15-scam-shield.braylo777.workers.dev

## What
Paste a text, email or DM and get `scam` / `legit`, a calibrated probability, and the signals that fired (link, money, urgency, shortcode, credentials). Runs entirely inside a Cloudflare Worker.

## Why
Scam texts and phishing DMs keep getting better as AI writes them. A small, explainable, cheap classifier at the edge is the first filter: it answers in milliseconds with no per-request LLM cost.

## How it was built
- **Data:** UCI SMS Spam Collection (5,574 labeled messages, 747 spam), in `data/SMSSpamCollection`.
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
```bash
python3 -m pytest                       # Python tests (no model calls)
cd worker && cargo test                 # Rust tests
python3 evals/train.py                  # retrain; writes data/model.json and evals/results.json
cd worker && worker-build --release     # build the Worker (wasm)
wrangler deploy                         # free tier
```
Tests: Python 6 test functions, Rust 2. The old JavaScript, the Python and the Rust code give identical labels, probabilities, signals, features and metrics on 412 texts (400 real SMS messages plus Unicode and edge cases). Retraining in Python reproduces the JavaScript-trained model exactly (same bias and all 8,159 non-zero weights), so the reported results stand.

The previous JavaScript version is on the `legacy-js` branch and the `pre-pycrust` tag. Embedding endpoint `/api/classify-embed` still reports an untrained head (503), as before.
