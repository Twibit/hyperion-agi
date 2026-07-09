//! The ripgrep-class walker + matcher, wrapped as a `Limb`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use grep_matcher::Matcher;
use grep_regex::RegexMatcher;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;
use ignore::WalkBuilder;

use crate::cortex::MotorCommand;
use crate::error::{TkError, TkResult};
use crate::intent::Outcome;
use crate::limb::Limb;
use crate::mirror::Observer;
use crate::search::{score_hits, SearchHit, SearchResults};

/// The file-search limb — the software eye.
///
/// Rooted at a specific directory (its "receptive field"). Concurrency-safe:
/// the internal walker is stateless per-call, so the limb is cheap to share.
#[derive(Debug, Clone)]
pub struct SearchLimb {
    name: String,
    root: PathBuf,
    observer: Observer,
    max_file_bytes: u64,
}

impl SearchLimb {
    /// Root the eye at `root`.
    pub fn rooted(root: impl AsRef<Path>) -> Self {
        Self {
            name: "search-eye".to_string(),
            root: root.as_ref().to_path_buf(),
            observer: Observer::new("search-eye"),
            max_file_bytes: 4 * 1024 * 1024, // 4 MiB per file default
        }
    }

    /// Set the per-file byte cap; files above are skipped.
    pub fn with_max_file_bytes(mut self, n: u64) -> Self {
        self.max_file_bytes = n;
        self
    }

    /// Set a custom limb name (used in tracing).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self.observer = Observer::new(self.name.clone());
        self
    }

    fn do_search(
        &self,
        query: &str,
        pattern: Option<&str>,
        top_k: usize,
    ) -> TkResult<SearchResults> {
        let matcher =
            RegexMatcher::new_line_matcher(query).map_err(|e| TkError::Pattern(e.to_string()))?;
        let path_re: Option<regex::Regex> = match pattern {
            Some(p) => Some(regex::Regex::new(p).map_err(|e| TkError::Pattern(e.to_string()))?),
            None => None,
        };

        let hits = Arc::new(Mutex::new(Vec::<SearchHit>::new()));
        let files_scanned = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_bytes = self.max_file_bytes;

        let walker = WalkBuilder::new(&self.root)
            .standard_filters(true) // .gitignore, hidden, parents
            .follow_links(false)
            .build_parallel();

        walker.run(|| {
            let matcher = matcher.clone();
            let path_re = path_re.clone();
            let hits = Arc::clone(&hits);
            let files_scanned = Arc::clone(&files_scanned);
            Box::new(move |entry| {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => return ignore::WalkState::Continue,
                };
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    return ignore::WalkState::Continue;
                }
                let path = entry.path();
                if let Some(re) = &path_re {
                    if !re.is_match(&path.to_string_lossy()) {
                        return ignore::WalkState::Continue;
                    }
                }
                if let Ok(md) = entry.metadata() {
                    if md.len() > max_bytes {
                        return ignore::WalkState::Continue;
                    }
                }
                files_scanned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let mut local_hits: Vec<SearchHit> = Vec::new();
                let mut searcher = Searcher::new();
                let _ = searcher.search_path(
                    &matcher,
                    path,
                    UTF8(|lnum, line| {
                        let mut excerpt = line.trim_end().to_string();
                        if excerpt.len() > 512 {
                            excerpt.truncate(512);
                        }
                        if matcher.is_match(line.as_bytes()).unwrap_or(false) {
                            local_hits.push(SearchHit {
                                path: path.to_path_buf(),
                                line: lnum as usize,
                                excerpt,
                                score: 0.0,
                            });
                        }
                        Ok(true)
                    }),
                );

                if !local_hits.is_empty() {
                    if let Ok(mut guard) = hits.lock() {
                        guard.extend(local_hits);
                    }
                }
                ignore::WalkState::Continue
            })
        });

        let raw = Arc::try_unwrap(hits)
            .map_err(|_| TkError::plan("search hits arc still shared"))?
            .into_inner()
            .map_err(|_| TkError::plan("search hits mutex poisoned"))?;
        let total = raw.len();
        let scored = score_hits(raw, query, top_k);
        Ok(SearchResults {
            total_matches: total,
            files_scanned: files_scanned.load(std::sync::atomic::Ordering::Relaxed),
            hits: scored,
        })
    }
}

#[async_trait]
impl Limb for SearchLimb {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, cmd: MotorCommand) -> TkResult<Outcome> {
        self.observer.about_to_act(&cmd);
        match cmd {
            MotorCommand::Search {
                query,
                pattern,
                top_k,
            } => {
                // Search is CPU-bound; run on the blocking pool.
                let this = self.clone();
                let res = tokio::task::spawn_blocking(move || {
                    this.do_search(&query, pattern.as_deref(), top_k)
                })
                .await
                .map_err(|e| TkError::plan(format!("search task joined: {e}")))??;
                Ok(Outcome::SearchHits { results: res })
            }
            other => Err(TkError::plan(format!(
                "SearchLimb cannot execute {other:?}; route to FileLimb"
            ))),
        }
    }
}
