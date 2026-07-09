//! Ranking layer for search hits.
//!
//! We score each *raw* hit independently, then normalise across the batch.
//! Design intent: the scorer has no I/O, no allocation past what's inherent
//! in `Vec`, and is fully deterministic given the same input.

use std::path::Path;

use crate::search::SearchHit;

/// Weights (chosen empirically; tune per workload).
const W_FILENAME_HIT: f32 = 3.0;
const W_INVERSE_DEPTH: f32 = 1.5;
const W_MATCH_COUNT_LOG: f32 = 1.0;

/// Rank a batch of raw hits in-place and return the top-k.
///
/// * `hits` is consumed and re-sorted; final scores are written back in.
/// * `query` is the substring we searched for — used for filename hits.
/// * `top_k` is the cap on the returned length.
pub fn score_hits(mut hits: Vec<SearchHit>, query: &str, top_k: usize) -> Vec<SearchHit> {
    // Per-file match counts.
    let mut counts: ahash::AHashMap<std::path::PathBuf, usize> = ahash::AHashMap::new();
    for h in &hits {
        *counts.entry(h.path.clone()).or_insert(0) += 1;
    }

    let q_lower = query.to_ascii_lowercase();
    let mut max_score = f32::MIN_POSITIVE;

    for h in &mut hits {
        let count = *counts.get(&h.path).unwrap_or(&1) as f32;
        let depth = path_depth(&h.path) as f32;
        let filename_hit = h
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_ascii_lowercase().contains(&q_lower))
            .unwrap_or(false);

        let raw = W_MATCH_COUNT_LOG * (1.0 + count).ln()
            + W_INVERSE_DEPTH * (1.0 / (1.0 + depth))
            + if filename_hit { W_FILENAME_HIT } else { 0.0 };

        h.score = raw;
        if raw > max_score {
            max_score = raw;
        }
    }

    // Normalise into [0, 1].
    if max_score > 0.0 {
        for h in &mut hits {
            h.score /= max_score;
        }
    }

    // Sort descending by score, tie-break by (path, line) for determinism.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
    hits.truncate(top_k);
    hits
}

fn path_depth(p: &Path) -> usize {
    p.components().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn hit(path: &str, line: usize) -> SearchHit {
        SearchHit {
            path: PathBuf::from(path),
            line,
            excerpt: String::new(),
            score: 0.0,
        }
    }

    #[test]
    fn shallower_paths_outrank_deeper() {
        let hits = vec![hit("a.rs", 1), hit("deep/nested/path/b.rs", 1)];
        let scored = score_hits(hits, "x", 2);
        assert_eq!(scored[0].path, PathBuf::from("a.rs"));
    }

    #[test]
    fn filename_hit_dominates() {
        let hits = vec![
            hit("readme.txt", 100),
            hit("query.rs", 1),
            hit("other.rs", 1),
        ];
        let scored = score_hits(hits, "query", 3);
        assert_eq!(scored[0].path, PathBuf::from("query.rs"));
    }

    #[test]
    fn top_k_truncates() {
        let hits: Vec<_> = (0..10).map(|i| hit(&format!("f{i}.rs"), 1)).collect();
        let scored = score_hits(hits, "x", 3);
        assert_eq!(scored.len(), 3);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(score_hits(vec![], "x", 5).is_empty());
    }
}
