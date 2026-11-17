"""Trains the hashed-ngram logistic regression on a seeded stratified 80/20 split.
Writes data/model.json + evals/results.json. Usage: python3 evals/train.py [--out DIR]"""
import json, math, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from agent15.features import DIM, featurize, sigmoid  # noqa: E402
from agent15.model import metrics, to_fixed  # noqa: E402

out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else root
rows = []
for l in (root / "data" / "SMSSpamCollection").read_text(encoding="utf-8", errors="replace").strip().split("\n"):
    a, *r = l.split("\t")
    rows.append({"y": 1 if a == "spam" else 0, "t": "\t".join(r)})
s = [42]
def rnd():
    s[0] = ((((s[0] * 1103515245) & 0xFFFFFFFF) + 12345) & 0xFFFFFFFF)  # Math.imul(s, a) + c >>> 0
    return s[0] / 2 ** 32
train, test = [], []
for c in (0, 1):
    g = [x[1] for x in sorted(([rnd(), r] for r in rows if r["y"] == c), key=lambda x: x[0])]
    k = round(len(g) * 0.8)
    train += g[:k]; test += g[k:]
X = [featurize(r["t"]) for r in train]
w = [0.0] * DIM; b = 0.0; lr, l2, E = 0.5, 1e-4, 30
for _ in range(E):
    order = [x[1] for x in sorted(([rnd(), i] for i in range(len(X))), key=lambda x: x[0])]
    for i in order:
        z = b
        for k, v in X[i]:
            z += w[k] * v
        g = sigmoid(z) - train[i]["y"]
        b -= lr * g
        for k, v in X[i]:
            w[k] -= lr * (g * v + l2 * w[k])
model = {"b": to_fixed(b, 5), "w": {str(k): to_fixed(v, 4) for k, v in enumerate(w) if abs(v) > 1e-3}}
(out_dir / "data").mkdir(exist_ok=True)
(out_dir / "data" / "model.json").write_text(json.dumps(model, separators=(",", ":")))
def sc(r):
    z = model["b"]
    for k, v in featurize(r["t"]):
        z += model["w"].get(str(k), 0) * v
    return sigmoid(z)
tm = metrics([r["y"] for r in test], [sc(r) for r in test])
tr = metrics([r["y"] for r in train], [sc(r) for r in train])
res = {"dataset": "UCI SMS Spam Collection (5,574 msgs, 747 spam)", "split": "stratified 80/20, seed 42", "n_train": len(train), "n_test": len(test), "epochs": E,
       "hashed_ngram_lr": {"test": tm, "train": {k: tr[k] for k in ("precision", "recall", "f1", "roc_auc")}},
       "embedding_head_bge_base": "not yet run (Workers AI free daily quota exhausted until 7 PM CT Oct 28)", "run_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")}
(out_dir / "evals").mkdir(exist_ok=True)
(out_dir / "evals" / "results.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: tm[k] for k in ("n", "precision", "recall", "f1", "roc_auc", "ece_10bin", "brier", "tp", "fp", "fn", "tn")}))
