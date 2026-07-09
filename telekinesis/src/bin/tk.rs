//! `tk` — command-line demo for the telekinesis crate.
//!
//! Not intended as a production tool; it exists so you can see the whole
//! sensorimotor loop from a shell.
//!
//! Examples:
//!
//! ```bash
//! tk search "fn main" --path ./src --top 10
//! tk reach --op read --path ./Cargo.toml
//! ```

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use telekinesis::prelude::*;

#[derive(Debug, Parser)]
#[command(name = "tk", version, about = "Telekinesis CLI demo")]
struct Cli {
    /// Increase log verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a ranked file search under `--path`.
    Search {
        /// Regex or literal to search for.
        query: String,
        /// Root of the search.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Optional regex on file paths.
        #[arg(long)]
        pattern: Option<String>,
        /// Top-k cap.
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// Reach with the file limb (read / delete).
    Reach {
        /// Operation: read | delete
        #[arg(long)]
        op: String,
        /// Target path.
        #[arg(long)]
        path: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| level.into()),
        )
        .with_target(false)
        .init();

    let mut ns = NervousSystem::new().await?;

    match cli.cmd {
        Command::Search {
            query,
            path,
            pattern,
            top,
        } => {
            let eye = ns.grow_limb(SearchLimb::rooted(&path)).await;
            let out = eye
                .reach(Intent::Search {
                    query: query.clone(),
                    pattern,
                    top_k: top,
                })
                .await?;
            let res = out
                .into_search()
                .ok_or_else(|| anyhow::anyhow!("unexpected outcome"))?;
            println!(
                "scanned {} files, {} raw hits, top {}:",
                res.files_scanned,
                res.total_matches,
                res.hits.len()
            );
            for (i, h) in res.hits.iter().enumerate() {
                println!(
                    "  {:>2}. [{:0.3}] {}:{}  {}",
                    i + 1,
                    h.score,
                    h.path.display(),
                    h.line,
                    h.excerpt
                );
            }
        }
        Command::Reach { op, path } => {
            let hand = ns.grow_limb(FileLimb::new()).await;
            let intent = match op.as_str() {
                "read" => Intent::Read { path },
                "delete" => Intent::Delete { path },
                other => anyhow::bail!("unknown op: {other}"),
            };
            match hand.reach(intent).await? {
                Outcome::Bytes { data: b } => {
                    println!("{}", String::from_utf8_lossy(&b));
                }
                Outcome::Deleted => println!("deleted"),
                other => println!("{:?}", other),
            }
        }
    }
    Ok(())
}
