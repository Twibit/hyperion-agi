//! End-to-end example: search for TODOs in Rust files, annotate the first
//! hit in a temporary sandbox, verify via reafference, and print the loop
//! diagnostics.

use std::io::Write;

use telekinesis::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("telekinesis=debug,warn")
        .init();

    // Build a tiny sandbox so the example is self-contained.
    let dir = tempfile::tempdir()?;
    let sample = dir.path().join("sample.rs");
    let mut f = std::fs::File::create(&sample)?;
    writeln!(f, "fn main() {{")?;
    writeln!(f, "    // TODO: implement telekinesis")?;
    writeln!(f, "    println!(\"hello\");")?;
    writeln!(f, "}}")?;
    drop(f);

    let mut ns = NervousSystem::new().await?;
    let eye = ns.grow_limb(SearchLimb::rooted(dir.path())).await;
    let hand = ns.grow_limb(FileLimb::new()).await;

    let hits = eye
        .reach(Intent::Search {
            query: "TODO".into(),
            pattern: Some(r"\.rs$".into()),
            top_k: 5,
        })
        .await?
        .into_search()
        .expect("search outcome")
        .into_top();

    println!("found {} hit(s)", hits.len());
    for h in &hits {
        println!("  {}:{}  {}", h.path.display(), h.line, h.excerpt);
        hand.reach(Intent::Annotate {
            path: h.path.clone(),
            line: h.line,
            note: "flagged by telekinesis demo".into(),
        })
        .await?;
    }

    let final_bytes = hand
        .reach(Intent::Read {
            path: sample.clone(),
        })
        .await?
        .into_bytes()
        .unwrap();
    println!("--- final file ---\n{}", String::from_utf8_lossy(&final_bytes));
    Ok(())
}
