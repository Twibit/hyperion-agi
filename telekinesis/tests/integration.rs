//! End-to-end integration tests exercising the full sensorimotor loop.

use std::io::Write;

use telekinesis::prelude::*;

#[tokio::test]
async fn read_write_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("hello.txt");

    let mut ns = NervousSystem::new().await.unwrap();
    let hand = ns.grow_limb(FileLimb::new()).await;

    hand.reach(Intent::Write {
        path: p.clone(),
        data: b"hello world".to_vec(),
    })
    .await
    .unwrap();

    let bytes = hand
        .reach(Intent::Read { path: p.clone() })
        .await
        .unwrap()
        .into_bytes()
        .unwrap();
    assert_eq!(bytes, b"hello world");
}

#[tokio::test]
async fn annotate_inserts_comment() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("code.rs");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, "fn a() {{}}").unwrap();
    writeln!(f, "fn b() {{}}").unwrap();
    drop(f);

    let mut ns = NervousSystem::new().await.unwrap();
    let hand = ns.grow_limb(FileLimb::new()).await;

    hand.reach(Intent::Annotate {
        path: p.clone(),
        line: 2,
        note: "watch this".into(),
    })
    .await
    .unwrap();

    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("// TELEKINESIS: watch this"));
    // The annotation should appear *before* line 2's original content.
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[1], "// TELEKINESIS: watch this");
    assert_eq!(lines[2], "fn b() {}");
}

#[tokio::test]
async fn delete_and_read_missing() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("victim.txt");
    std::fs::write(&p, b"data").unwrap();

    let mut ns = NervousSystem::new().await.unwrap();
    let hand = ns.grow_limb(FileLimb::new()).await;

    hand.reach(Intent::Delete { path: p.clone() })
        .await
        .unwrap();
    assert!(!p.exists());

    let err = hand.reach(Intent::Read { path: p }).await.unwrap_err();
    assert!(matches!(err, TkError::Effector(_)));
}

#[tokio::test]
async fn search_finds_and_ranks() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..3 {
        let p = dir.path().join(format!("f{i}.rs"));
        std::fs::write(&p, format!("fn main() {{ println!(\"needle {i}\"); }}\n")).unwrap();
    }
    // A file whose *name* matches the query — should rank first.
    let p = dir.path().join("needle.rs");
    std::fs::write(&p, "fn other() {}\n").unwrap();
    // fabricate a matching line so scorer has something to rank
    std::fs::write(&p, "// needle in filename\n").unwrap();

    let mut ns = NervousSystem::new().await.unwrap();
    let eye = ns.grow_limb(SearchLimb::rooted(dir.path())).await;

    let res = eye
        .reach(Intent::Search {
            query: "needle".into(),
            pattern: Some(r"\.rs$".into()),
            top_k: 10,
        })
        .await
        .unwrap()
        .into_search()
        .unwrap();

    assert!(res.total_matches >= 3);
    assert!(!res.hits.is_empty());
    // The filename-hit file should be the top result.
    assert_eq!(
        res.hits[0].path.file_name().unwrap().to_string_lossy(),
        "needle.rs"
    );
    // Scores are normalised to [0, 1].
    for h in &res.hits {
        assert!(h.score >= 0.0 && h.score <= 1.0);
    }
}

#[tokio::test]
async fn search_limb_rejects_write() {
    let dir = tempfile::tempdir().unwrap();
    let mut ns = NervousSystem::new().await.unwrap();
    let eye = ns.grow_limb(SearchLimb::rooted(dir.path())).await;
    let err = eye
        .reach(Intent::Write {
            path: dir.path().join("x"),
            data: vec![1, 2, 3],
        })
        .await
        .unwrap_err();
    // The search limb cannot execute a WriteAll command; it should surface
    // as a motor-plan / execution failure.
    assert!(matches!(err, TkError::MotorPlan(_)));
}
