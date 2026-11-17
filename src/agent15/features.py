"""Model-free features: hashed word unigrams/bigrams + char 3-grams + a few scam signals.
Hashing runs over UTF-16 code units so it matches the JavaScript model weights exactly."""
import math, re

DIM = 8192
_JS_WS = re.compile("[\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]+")
_WORD = re.compile("[a-z0-9£$€]+")
A = re.ASCII
_URL = re.compile(r"https?://|www\.|\.(com|net|ly|co)\b", A)
_LONGNUM = re.compile(r"\b\d{5,}\b", A)
_MONEY = re.compile(r"[£$€]\s?\d")
_URGENT = re.compile(r"\b(urgent|verify|suspended|claim|winner|won|prize|act now|gift card|password|account)\b", A)


def units(s):
    b = s.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(b[i:i + 2], "little") for i in range(0, len(b), 2)]


def h_units(us):
    x = 2166136261
    for u in us:
        x = ((x ^ u) * 16777619) & 0xFFFFFFFF
    return x % DIM


def h(s):
    return h_units(units(s))


def sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0.0


def featurize(text):
    t = ("" if not text else str(text)).lower()
    w = _WORD.findall(t)
    f = {}
    def add(k, v=1):
        f[k] = f.get(k, 0) + v
    for i, x in enumerate(w):
        add(h("w:" + x))
        if i:
            add(h("b:" + w[i - 1] + "_" + x))
    c = units(" " + _JS_WS.sub(" ", t) + " ")
    pre = [ord("c"), ord(":")]
    for i in range(len(c) - 2):
        add(h_units(pre + c[i:i + 3]), 0.3)
    if _URL.search(t): add(h("sig:url"), 2)
    if _LONGNUM.search(t): add(h("sig:longnum"), 2)
    if _MONEY.search(t): add(h("sig:money"), 2)
    if _URGENT.search(t): add(h("sig:urgency"), 2)
    n = 0
    for v in f.values():
        n += v * v
    n = math.sqrt(n) or 1
    return [(k, v / n) for k, v in f.items()]


def score(model, text):
    z = model["b"]
    for k, v in featurize(text):
        z += model["w"].get(str(k), 0) * v
    return sigmoid(z)
