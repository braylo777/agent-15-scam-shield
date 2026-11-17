// Parity helper: same inputs as the legacy JS harness; prints JSON.
use agent_15_scam_shield::{features::featurize, model::*};
use serde_json::{json, Value};
fn main() {
    let t: Vec<String> = serde_json::from_str(&std::fs::read_to_string("/tmp/in15.json").unwrap()).unwrap();
    let y: Vec<i64> = (0..t.len()).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let p: Vec<f64> = t.iter().map(|x| classify(x)["p_scam"].as_f64().unwrap()).collect();
    let f: Vec<Value> = t[t.len() - 12..].iter().map(|x| Value::Array(featurize(x).into_iter().take(8).map(|(k, v)| json!([k, v])).collect())).collect();
    let o = json!({"c": t.iter().map(|x| classify(x)).collect::<Vec<_>>(), "f": f, "m": metrics(&y, &p), "m2": metrics(&[1, 0, 1, 0, 1], &[0.5, 0.5, 0.2, 0.9, 0.5])});
    println!("{}", serde_json::to_string(&o).unwrap());
}
