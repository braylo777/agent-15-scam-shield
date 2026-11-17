//! Metrics, the trained model and the classifier.
use crate::features::{featurize, sigmoid, DIM};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

const MODEL_JSON: &str = include_str!("../data/model.json");
pub const THRESHOLD: f64 = 0.5;

pub struct Model {
    pub b: f64,
    pub w: Vec<f64>,
}
pub fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| {
        let v: Value = serde_json::from_str(MODEL_JSON).unwrap();
        let mut w = vec![0.0; DIM as usize];
        for (k, x) in v["w"].as_object().unwrap() {
            w[k.parse::<usize>().unwrap()] = x.as_f64().unwrap();
        }
        Model { b: v["b"].as_f64().unwrap(), w }
    })
}
pub fn score(text: &str) -> f64 {
    let m = model();
    let mut z = m.b;
    for (k, v) in featurize(text) {
        z += m.w[k as usize] * v;
    }
    sigmoid(z)
}

/// JS Number(x.toFixed(d)): exact decimal expansion, ties round up.
pub fn to_fixed(x: f64, d: usize) -> f64 {
    let s = format!("{:.60}", x.abs());
    let (int, frac) = s.split_once('.').unwrap();
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes().take(d)).map(|b| b - b'0').collect();
    if frac.as_bytes().get(d).map_or(0, |b| b - b'0') >= 5 {
        let mut i = digits.len();
        loop {
            if i == 0 { digits.insert(0, 1); break; }
            i -= 1;
            if digits[i] == 9 { digits[i] = 0 } else { digits[i] += 1; break; }
        }
    }
    let n = digits.len();
    let (a, b) = digits.split_at(n - d);
    format!("{}{}.{}", if x < 0.0 { "-" } else { "" }, a.iter().map(|x| x.to_string()).collect::<String>(), b.iter().map(|x| x.to_string()).collect::<String>()).parse().unwrap_or(x)
}

fn signals() -> &'static Vec<(&'static str, Regex)> {
    static S: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    S.get_or_init(|| {
        let b = |p: &str| Regex::new(&format!("(?i){}", p.replace(r"\b", r"(?-u:\b)"))).unwrap();
        vec![("link", b(r"https?://|www\.")), ("money", b(r"[£$€]\s?[0-9]|\bprize|cash\b")), ("urgency", b(r"\burgent|act now|expires?|suspended|verify\b")), ("phone/shortcode", Regex::new(r"(?-u:\b)[0-9]{5,}(?-u:\b)").unwrap()), ("credentials", b(r"\bpassword|pin|ssn|login|account\b"))]
    })
}

pub fn classify(text: &str) -> Value {
    let p = score(text);
    json!({"label": if p >= THRESHOLD { "scam" } else { "legit" }, "p_scam": to_fixed(p, 4), "signals": signals().iter().filter(|(_, r)| r.is_match(text)).map(|(n, _)| *n).collect::<Vec<_>>(), "model": "hashed-ngram logistic regression (trained on box)"})
}

/// Embedding head (bge-base + logistic). Needs Workers AI; head is trained once quota resets.
pub fn classify_embed() -> Value {
    json!({"error": "embedding head not yet trained (Workers AI quota reset pending)"})
}

fn js_num(v: f64) -> String {
    if v == v.trunc() { format!("{}", v as i64) } else { format!("{}", v) }
}

pub fn metrics(y: &[i64], p: &[f64]) -> Value {
    let thr = 0.5;
    let (mut tp, mut fp, mut fnn, mut tn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (t, v) in y.iter().zip(p) {
        let k = *v >= thr;
        if k && *t == 1 { tp += 1.0 } else if k { fp += 1.0 } else if *t == 1 { fnn += 1.0 } else { tn += 1.0 }
    }
    let nz = |x: f64| if x == 0.0 { 1.0 } else { x };
    let precision = tp / nz(tp + fp);
    let recall = tp / nz(tp + fnn);
    let f1 = (2.0 * precision * recall) / nz(precision + recall);
    let mut idx: Vec<(f64, i64)> = p.iter().copied().zip(y.iter().copied()).collect();
    idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut r, mut pos, mut i) = (0.0f64, 0.0f64, 0usize);
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && idx[j].0 == idx[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            if idx[k].1 == 1 { r += avg; pos += 1.0; }
        }
        i = j;
    }
    let neg = idx.len() as f64 - pos;
    let auc = (r - (pos * (pos + 1.0)) / 2.0) / nz(pos * neg);
    let mut bins = [[0.0f64; 3]; 10];
    for (v, t) in p.iter().zip(y) {
        let b = &mut bins[((v * 10.0).floor() as usize).min(9)];
        b[0] += 1.0; b[1] += v; b[2] += *t as f64;
    }
    let mut ece = 0.0;
    for [n, s, o] in bins {
        ece += if n > 0.0 { (n / p.len() as f64) * (s / n - o / n).abs() } else { 0.0 };
    }
    let brier = p.iter().zip(y).map(|(v, t)| (v - *t as f64).powi(2)).sum::<f64>() / p.len() as f64;
    let rel: Vec<Value> = bins.iter().enumerate().map(|(k, [n, s, o])| json!({"bin": format!("{}-{}", js_num(k as f64 / 10.0), js_num((k + 1) as f64 / 10.0)), "n": *n as i64, "mean_pred": if *n > 0.0 { json!(s / n) } else { Value::Null }, "frac_scam": if *n > 0.0 { json!(o / n) } else { Value::Null }})).collect();
    json!({"n": y.len(), "tp": tp as i64, "fp": fp as i64, "fn": fnn as i64, "tn": tn as i64, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc, "ece_10bin": ece, "brier": brier, "reliability": rel})
}

/// Pure router: (status, json); status 0 = serve the page. `limited` runs only for the embed route, after validation.
pub async fn handle<L: AsyncFn() -> bool>(method: &str, path: &str, body: &Value, limited: L) -> (u16, Value) {
    if method == "POST" && (path == "/api/classify" || path == "/api/classify-embed") {
        let text = body.get("text").and_then(|v| v.as_str());
        let Some(t) = text.filter(|t| !t.trim().is_empty() && t.encode_utf16().count() <= 4000) else { return (400, json!({"error": "text required (<=4000 chars)"})) };
        if path == "/api/classify" {
            return (200, classify(t));
        }
        if limited().await {
            return (429, json!({"error": "Rate limit: try again in a minute."}));
        }
        return (503, classify_embed());
    }
    if path == "/api/health" {
        return (200, json!({"ok": true}));
    }
    (0, Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::featurize;
    use futures::executor::block_on;

    #[test]
    fn classify_and_features() {
        assert_eq!(classify("WINNER!! You have won a £1000 prize. Call 09061701461 to claim now")["label"], "scam");
        assert_eq!(classify("ok see you at dinner tonight, I'll bring the kids")["label"], "legit");
        let r = classify("Verify your account at http://x.co urgent");
        let p = r["p_scam"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p) && r["signals"].as_array().unwrap().contains(&json!("link")));
        assert!((featurize("hello world hello").iter().map(|(_, v)| v * v).sum::<f64>() - 1.0).abs() < 1e-9);
        let r = classify("🎉🎉 héllo ñandú 你好");
        assert!((0.0..=1.0).contains(&r["p_scam"].as_f64().unwrap()));
        let _ = classify(&"ok ".repeat(50000));
        let t = "URGENT claim your reward now";
        assert_eq!(classify(t)["p_scam"], classify(t)["p_scam"]);
    }
    #[test]
    fn metrics_and_api() {
        let m = metrics(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9]);
        assert_eq!((m["roc_auc"].as_f64(), m["f1"].as_f64()), (Some(1.0), Some(1.0)));
        assert_eq!(metrics(&[1, 0], &[0.1, 0.9])["roc_auc"].as_f64(), Some(0.0));
        assert!(metrics(&[1, 0], &[1.0, 0.0])["ece_10bin"].as_f64().unwrap() < 1e-9);
        let go = |p: &str, b: Value, l: bool| block_on(handle("POST", p, &b, async move || l));
        assert_eq!(go("/api/classify", json!({"text": ""}), false).0, 400);
        assert_eq!(go("/api/classify", json!("{not json"), false).0, 400);
        assert_eq!(go("/api/classify", json!({"text": "Free entry! Text WIN to 80086"}), false).1["label"], "scam");
        assert_eq!(go("/api/classify-embed", json!({"text": "hi"}), false).0, 503);
        assert_eq!(go("/api/classify-embed", json!({"text": "hi"}), true).0, 429);
        assert_eq!(block_on(handle("GET", "/api/health", &json!({}), async || false)).1, json!({"ok": true}));
        assert_eq!(block_on(handle("GET", "/", &json!({}), async || false)).0, 0);
    }
}
