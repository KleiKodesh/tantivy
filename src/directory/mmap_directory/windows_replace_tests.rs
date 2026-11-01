//! A commit replaces `meta.json` while another handle holds it open.

use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::collector::Count;
use crate::query::AllQuery;
use crate::schema::{Schema, STRING};
use crate::{doc, Index, IndexWriter};

const FILE_SHARE_READ: u32 = 1;
const FILE_SHARE_WRITE: u32 = 2;
const FILE_SHARE_DELETE: u32 = 4;

fn hold(path: &Path, share_mode: u32) -> File {
    OpenOptions::new()
        .read(true)
        .share_mode(share_mode)
        .open(path)
        .unwrap()
}

fn num_docs(index: &Index) -> usize {
    let reader = index.reader().unwrap();
    reader.reload().unwrap();
    reader.searcher().search(&AllQuery, &Count).unwrap()
}

#[test]
fn commit_replaces_meta_held_open_with_share_delete() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("t", STRING);
    let index = Index::create_in_dir(dir.path(), schema_builder.build()).unwrap();
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    let held = hold(
        &dir.path().join("meta.json"),
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    );
    writer.add_document(doc!(field => "a")).unwrap();
    writer.commit().unwrap();
    drop(held);
    assert_eq!(num_docs(&index), 1);
}

#[test]
fn commit_blocked_by_meta_reader_fails_then_retry_succeeds() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("t", STRING);
    let index = Index::create_in_dir(dir.path(), schema_builder.build()).unwrap();
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    // No FILE_SHARE_DELETE: the file cannot be replaced while this handle is open.
    let held = hold(&dir.path().join("meta.json"), FILE_SHARE_READ);
    writer.add_document(doc!(field => "a")).unwrap();
    let start = Instant::now();
    let err = writer.commit().unwrap_err();
    let elapsed = start.elapsed();
    assert!(
        err.to_string().contains("held open by another process"),
        "{err}"
    );
    assert!(elapsed >= Duration::from_millis(400), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    drop(held);
    // The pending document survives the failed commit.
    writer.commit().unwrap();
    assert_eq!(num_docs(&index), 1);
    writer.add_document(doc!(field => "b")).unwrap();
    writer.commit().unwrap();
    assert_eq!(num_docs(&index), 2);
}
