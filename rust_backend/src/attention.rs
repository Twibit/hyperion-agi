use pyo3::prelude::*;

/// Sparse dot-product attention — EXACT match to rust_backend_shim.py semantics.
/// `keys` are pre-computed precision/dot-product scores; we softmax them.
/// If `values` is empty, returns the attention weights (list of floats).
/// Otherwise returns the weighted sum as a single-element list (matches the
/// shim's 1-D branch closely; no caller passes non-empty values). This is
/// what `claustrum.py` relies on (it passes `values=[]`).
#[pyfunction]
pub fn sparse_attention(keys: Vec<f64>, values: Vec<f64>, temperature: f64) -> PyResult<Vec<f64>> {
    if keys.is_empty() {
        return Ok(Vec::new());
    }
    let t = temperature.max(1e-6);
    let mut logits: Vec<f64> = keys.iter().map(|&k| k / t).collect();
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for l in &mut logits {
        *l -= max_logit;
    }
    let mut w: Vec<f64> = logits.iter().map(|&l| l.exp()).collect();
    let sum: f64 = w.iter().sum();
    for x in &mut w {
        *x /= sum;
    }

    if values.is_empty() {
        return Ok(w);
    }

    // 1-D weighted sum branch (matches shim's `float((w * v).sum())`).
    let acc: f64 = w.iter().zip(values.iter()).map(|(wi, &v)| wi * v).sum();
    Ok(vec![acc])
}
