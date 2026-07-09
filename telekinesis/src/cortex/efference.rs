//! Efference copy — the predicted-outcome snapshot.
//!
//! Named after the *Efferenzkopie* of von Holst & Mittelstaedt (1950).
//! Whenever a mutating command is about to be dispatched, the cortex takes
//! a snapshot of the current state of the affected resource. The
//! [`Proprioceptor`](crate::proprio::Proprioceptor) later compares reality
//! against this snapshot; on divergence, the actor rolls back.
//!
//! The buffer is LRU-bounded (default 1024 entries) so that on high-throughput
//! agents we cannot leak memory. This is a real bug we've seen in naive
//! "undo stack" implementations of the same idea.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lru::LruCache;
use uuid::Uuid;

/// A single predicted-state snapshot.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The path whose state was snapshotted.
    pub path: PathBuf,
    /// The bytes that existed before mutation (`None` if the file didn't
    /// exist — used for delete-and-restore semantics).
    pub prior_bytes: Option<Vec<u8>>,
    /// A short marker (size + first 8 bytes hash proxy) for cheap comparison
    /// by the proprioceptor when it doesn't want to re-read the full content.
    pub marker: String,
}

impl Snapshot {
    /// Build a marker string from bytes. Simple non-cryptographic proxy.
    pub fn marker_for(bytes: Option<&[u8]>) -> String {
        match bytes {
            None => "∅".to_string(),
            Some(b) => {
                let len = b.len();
                // Use ahash for a stable, fast, non-cryptographic digest.
                let hash = ahash::RandomState::with_seeds(1, 2, 3, 4).hash_one(b);
                format!("{len}:{hash:016x}")
            }
        }
    }
}

/// The bounded, thread-safe store of pending efference copies.
#[derive(Debug)]
pub struct EfferenceCopy {
    inner: Mutex<LruCache<Uuid, Snapshot>>,
}

impl EfferenceCopy {
    /// Create a store with the given capacity. Falls back to `1` if `cap` is 0.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = NonZeroUsize::new(cap.max(1)).unwrap();
        Self {
            inner: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Default 1024-slot store.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Take a snapshot of a path *before* mutation.
    ///
    /// If the file does not exist, we still register a snapshot (with
    /// `prior_bytes: None`) so that on rollback we know to delete a
    /// spuriously-created file rather than restore old bytes.
    pub fn snapshot(&self, path: &Path) -> Uuid {
        let bytes = std::fs::read(path).ok();
        let marker = Snapshot::marker_for(bytes.as_deref());
        let id = Uuid::new_v4();
        let snap = Snapshot {
            path: path.to_path_buf(),
            prior_bytes: bytes,
            marker,
        };
        // Lock is intentionally short — we only touch the LRU.
        self.inner
            .lock()
            .expect("efference-copy mutex poisoned")
            .put(id, snap);
        id
    }

    /// Retrieve (without removing) a snapshot by id.
    pub fn peek(&self, id: Uuid) -> Option<Snapshot> {
        self.inner
            .lock()
            .expect("efference-copy mutex poisoned")
            .peek(&id)
            .cloned()
    }

    /// Take a snapshot back out (used on rollback).
    pub fn take(&self, id: Uuid) -> Option<Snapshot> {
        self.inner
            .lock()
            .expect("efference-copy mutex poisoned")
            .pop(&id)
    }

    /// Current number of live snapshots.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("efference-copy mutex poisoned")
            .len()
    }

    /// Whether the store holds no snapshots.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for EfferenceCopy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn snapshot_of_existing_file_captures_bytes() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, b"hello").unwrap();
        let ec = EfferenceCopy::new();
        let id = ec.snapshot(&p);
        let snap = ec.peek(id).unwrap();
        assert_eq!(snap.prior_bytes.as_deref(), Some(&b"hello"[..]));
        assert!(snap.marker.starts_with("5:"));
    }

    #[test]
    fn snapshot_of_missing_file_yields_none_bytes() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("missing");
        let ec = EfferenceCopy::new();
        let id = ec.snapshot(&p);
        let snap = ec.peek(id).unwrap();
        assert!(snap.prior_bytes.is_none());
        assert_eq!(snap.marker, "∅");
    }

    #[test]
    fn take_removes_snapshot() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, b"x").unwrap();
        let ec = EfferenceCopy::new();
        let id = ec.snapshot(&p);
        assert!(ec.take(id).is_some());
        assert!(ec.peek(id).is_none());
    }

    #[test]
    fn lru_bounds_memory() {
        let ec = EfferenceCopy::with_capacity(2);
        let dir = tempdir().unwrap();
        for i in 0..10 {
            let p = dir.path().join(format!("f{i}"));
            std::fs::write(&p, b"z").unwrap();
            ec.snapshot(&p);
        }
        assert_eq!(ec.len(), 2, "LRU should cap at capacity");
    }
}
