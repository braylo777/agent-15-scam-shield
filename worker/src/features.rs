//! Model-free features: hashed word unigrams/bigrams + char 3-grams + a few scam signals.
//! Hashing runs over UTF-16 code units so it matches the JavaScript model weights exactly.
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const DIM: u32 = 8192;
const JS_WS: &str = r"[\t\n\x0B\x0C\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}]+";

fn rx() -> &'static [Regex; 6] {
    static R: OnceLock<[Regex; 6]> = OnceLock::new();
    R.get_or_init(|| {
        let b = |p: &str| Regex::new(&p.replace(r"\b", r"(?-u:\b)")).unwrap();
        [Regex::new(JS_WS).unwrap(), Regex::new("[a-z0-9£$€]+").unwrap(), b(r"https?://|www\.|\.(com|net|ly|co)\b"), b(r"\b[0-9]{5,}\b"), Regex::new(r"[£$€]\s?[0-9]").unwrap(), b(r"\b(urgent|verify|suspended|claim|winner|won|prize|act now|gift card|password|account)\b")]
    })
}

pub fn h_units(us: impl Iterator<Item = u16>) -> u32 {
    let mut x: u32 = 2166136261;
    for u in us {
        x = (x ^ u as u32).wrapping_mul(16777619);
    }
    x % DIM
}
pub fn h(s: &str) -> u32 {
    h_units(s.encode_utf16())
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn featurize(text: &str) -> Vec<(u32, f64)> {
    let r = rx();
    let t = text.to_lowercase();
    let w: Vec<&str> = r[1].find_iter(&t).map(|m| m.as_str()).collect();
    let (mut order, mut idx): (Vec<(u32, f64)>, HashMap<u32, usize>) = (vec![], HashMap::new());
    let mut add = |k: u32, v: f64| match idx.get(&k) {
        Some(&i) => order[i].1 += v,
        None => {
            idx.insert(k, order.len());
            order.push((k, v));
        }
    };
    for (i, x) in w.iter().enumerate() {
        add(h(&format!("w:{x}")), 1.0);
        if i > 0 {
            add(h(&format!("b:{}_{}", w[i - 1], x)), 1.0);
        }
    }
    let c: Vec<u16> = format!(" {} ", r[0].replace_all(&t, " ")).encode_utf16().collect();
    let pre = [b'c' as u16, b':' as u16];
    for i in 0..c.len().saturating_sub(2) {
        add(h_units(pre.iter().copied().chain(c[i..i + 3].iter().copied())), 0.3);
    }
    if r[2].is_match(&t) { add(h("sig:url"), 2.0); }
    if r[3].is_match(&t) { add(h("sig:longnum"), 2.0); }
    if r[4].is_match(&t) { add(h("sig:money"), 2.0); }
    if r[5].is_match(&t) { add(h("sig:urgency"), 2.0); }
    let mut n = 0.0;
    for (_, v) in &order {
        n += v * v;
    }
    let n = if n.sqrt() == 0.0 { 1.0 } else { n.sqrt() };
    order.into_iter().map(|(k, v)| (k, v / n)).collect()
}
