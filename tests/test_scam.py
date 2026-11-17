import pytest
from agent15.model import classify, metrics, handle
from agent15.features import featurize


def test_flags_prize_scam_and_passes_normal():
    assert classify("WINNER!! You have won a £1000 prize. Call 09061701461 to claim now")["label"] == "scam"
    assert classify("ok see you at dinner tonight, I'll bring the kids")["label"] == "legit"


def test_probability_and_signals():
    r = classify("Verify your account at http://x.co urgent")
    assert 0 <= r["p_scam"] <= 1 and "link" in r["signals"]


def test_features_l2_normalised():
    assert abs(sum(v * v for _, v in featurize("hello world hello")) - 1) < 1e-9


def test_metrics():
    m = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1 and m["f1"] == 1
    assert metrics([1, 0], [0.1, 0.9])["roc_auc"] == 0
    assert metrics([1, 0], [1, 0])["ece_10bin"] < 1e-9


def test_api():
    assert handle("POST", "/api/classify", {"text": ""})[0] == 400
    assert handle("POST", "/api/classify", "{not json")[0] == 400
    assert handle("POST", "/api/classify", {"text": "Free entry! Text WIN to 80086"})[1]["label"] == "scam"
    assert handle("POST", "/api/classify-embed", {"text": "hi"})[0] == 503
    assert handle("POST", "/api/classify-embed", {"text": "hi"}, limited=lambda: True)[0] == 429
    assert handle("GET", "/api/health")[1] == {"ok": True} and handle("GET", "/")[0] == 0


def test_edges():
    r = classify("🎉🎉 héllo ñandú 你好")
    assert 0 <= r["p_scam"] <= 1
    classify("ok " * 50000)
    t = "URGENT claim your reward now"
    assert classify(t)["p_scam"] == classify(t)["p_scam"]
