export function metrics(y, p, thr = 0.5) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  y.forEach((t, i) => { const k = p[i] >= thr; if (k && t) tp++; else if (k) fp++; else if (t) fn++; else tn++; });
  const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1), f1 = (2 * precision * recall) / (precision + recall || 1);
  // ROC-AUC via rank statistic (ties averaged)
  const idx = p.map((v, i) => [v, y[i]]).sort((a, b) => a[0] - b[0]); let r = 0, pos = 0, i = 0;
  while (i < idx.length) { let j = i; while (j < idx.length && idx[j][0] === idx[i][0]) j++; const avg = (i + j + 1) / 2; for (let k = i; k < j; k++) if (idx[k][1]) { r += avg; pos++; } i = j; }
  const neg = idx.length - pos, auc = (r - (pos * (pos + 1)) / 2) / (pos * neg || 1);
  const bins = Array.from({ length: 10 }, () => [0, 0, 0]); p.forEach((v, i) => { const b = bins[Math.min(9, Math.floor(v * 10))]; b[0]++; b[1] += v; b[2] += y[i]; });
  const ece = bins.reduce((a, [n, s, o]) => a + (n ? (n / p.length) * Math.abs(s / n - o / n) : 0), 0);
  const brier = p.reduce((a, v, i) => a + (v - y[i]) ** 2, 0) / p.length;
  return { n: y.length, tp, fp, fn, tn, precision, recall, f1, roc_auc: auc, ece_10bin: ece, brier, reliability: bins.map(([n, s, o], k) => ({ bin: `${k / 10}-${(k + 1) / 10}`, n, mean_pred: n ? s / n : null, frac_scam: n ? o / n : null })) };
}
