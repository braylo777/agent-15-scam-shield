"""Metrics, the trained model and the classifier."""
import json, math, re
from pathlib import Path

from .features import featurize, score, sigmoid

MODEL_W = json.loads((Path(__file__).resolve().parents[2] / "data" / "model.json").read_text())
THRESHOLD = 0.5
A = re.ASCII
SIGNALS = [("link", re.compile(r"https?://|www\.", re.I | A)), ("money", re.compile(r"[£$€]\s?\d|\bprize|cash\b", re.I | A)),
           ("urgency", re.compile(r"\burgent|act now|expires?|suspended|verify\b", re.I | A)), ("phone/shortcode", re.compile(r"\b\d{5,}\b", A)),
           ("credentials", re.compile(r"\bpassword|pin|ssn|login|account\b", re.I | A))]


def to_fixed(x, d):
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def js_num(v):
    if v == int(v) and abs(v) < 1e21:
        return str(int(v))
    return repr(v)


def classify(text):
    p = score(MODEL_W, text)
    return {"label": "scam" if p >= THRESHOLD else "legit", "p_scam": to_fixed(p, 4), "signals": [n for n, r in SIGNALS if r.search(text)], "model": "hashed-ngram logistic regression (trained on box)"}


def classify_embed(head=None):
    """Embedding head (bge-base + logistic). Needs Workers AI; head is trained once quota resets."""
    return {"error": "embedding head not yet trained (Workers AI quota reset pending)"}


def metrics(y, p, thr=0.5):
    tp = fp = fn = tn = 0
    for t, v in zip(y, p):
        k = v >= thr
        if k and t: tp += 1
        elif k: fp += 1
        elif t: fn += 1
        else: tn += 1
    precision = tp / ((tp + fp) or 1)
    recall = tp / ((tp + fn) or 1)
    f1 = (2 * precision * recall) / ((precision + recall) or 1)
    idx = sorted(zip(p, y), key=lambda a: a[0])
    r = 0.0; pos = 0; i = 0
    while i < len(idx):
        j = i
        while j < len(idx) and idx[j][0] == idx[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        for k in range(i, j):
            if idx[k][1]:
                r += avg; pos += 1
        i = j
    neg = len(idx) - pos
    auc = (r - (pos * (pos + 1)) / 2) / ((pos * neg) or 1)
    bins = [[0, 0.0, 0] for _ in range(10)]
    for v, t in zip(p, y):
        b = bins[min(9, math.floor(v * 10))]
        b[0] += 1; b[1] += v; b[2] += t
    ece = 0.0
    for n, s, o in bins:
        ece += (n / len(p)) * abs(s / n - o / n) if n else 0
    brier = sum((v - t) ** 2 for v, t in zip(p, y)) / len(p)
    return {"n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc, "ece_10bin": ece, "brier": brier,
            "reliability": [{"bin": f"{js_num(k / 10)}-{js_num((k + 1) / 10)}", "n": n, "mean_pred": s / n if n else None, "frac_scam": o / n if n else None} for k, (n, s, o) in enumerate(bins)]}


def cap_text(t):
    return isinstance(t, str) and bool(t.strip()) and len(t.encode("utf-16-le")) // 2 <= 4000


def handle(method, path, body=None, limited=lambda: False):
    """Pure router: (status, json); status 0 = serve the page."""
    if method == "POST" and path in ("/api/classify", "/api/classify-embed"):
        text = body.get("text") if isinstance(body, dict) else None
        if not cap_text(text):
            return 400, {"error": "text required (<=4000 chars)"}
        if path == "/api/classify":
            return 200, classify(text)
        if limited():
            return 429, {"error": "Rate limit: try again in a minute."}
        return 503, classify_embed()
    if path == "/api/health":
        return 200, {"ok": True}
    return 0, None
