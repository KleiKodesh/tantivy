//! Reordering merge: merge segments into one segment in a doc order the caller chooses.
//!
//! [`IndexWriter::merge_reordered`](crate::IndexWriter::merge_reordered) writes the merged
//! segment in the order a [`ReorderPlan`] gives (any permutation of the alive docs, not
//! only the order of the index's sort field), and can replace the values of numeric fast
//! fields on the way (e.g. renumber the sort field so that the new order is sorted by it).
//! The merge publishes like any merge: the new segment replaces the merged ones atomically
//! when it is complete; a cancelled or failed merge leaves the index as it was.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::index::SegmentReader;
use crate::DocAddress;

/// The doc order and fast-field values of a reordering merge.
pub struct ReorderPlan {
    /// `new_to_old[new_doc]` = the doc's address in the merged segments; segment ordinals
    /// index the readers given to the plan function. Must hold every alive doc exactly once.
    pub new_to_old: Vec<DocAddress>,
    /// Fast fields whose values are replaced: `(field name, new value of a doc)`, the
    /// function called with the doc's address in the merged segments. The field must be a
    /// single-valued `u64` fast field. The values are computed while the column is written
    /// (never held at once), so the function is called several times per doc and must be
    /// cheap and deterministic.
    pub fast_field_values: Vec<(String, FastFieldValueFn)>,
}

/// The new value of a replaced fast field for a doc, given its address in the merged
/// segments (see [`ReorderPlan::fast_field_values`]).
pub type FastFieldValueFn = Box<dyn Fn(DocAddress) -> u64 + Send + Sync>;

/// Builds the plan from the readers of the segments being merged (alive docs only).
pub type ReorderPlanFn =
    Box<dyn FnOnce(&[SegmentReader]) -> crate::Result<ReorderPlan> + Send + 'static>;

/// What a reordering merge is doing (see [`ReorderControl::on_progress`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReorderStage {
    /// The plan function is running.
    Plan,
    /// Postings: `done` of `total` terms (over all indexed fields).
    Postings,
    /// Doc store.
    Store,
    /// Fast fields.
    FastFields,
}

type ProgressFn = dyn Fn(ReorderStage, u64, u64) + Send + Sync;

/// Cancellation and progress of a reordering merge. Cheap to clone; shared with the
/// merge thread.
#[derive(Clone, Default)]
pub struct ReorderControl {
    cancelled: Arc<AtomicBool>,
    progress: Option<Arc<ProgressFn>>,
}

impl ReorderControl {
    /// A control that is not cancelled and reports no progress.
    pub fn new() -> ReorderControl {
        ReorderControl::default()
    }

    /// Uses the given flag for cancellation (e.g. one the application already shares).
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> ReorderControl {
        self.cancelled = flag;
        self
    }

    /// Calls `f(stage, done, total)` from the merge thread as the merge advances.
    pub fn on_progress(
        mut self,
        f: impl Fn(ReorderStage, u64, u64) + Send + Sync + 'static,
    ) -> ReorderControl {
        self.progress = Some(Arc::new(f));
        self
    }

    /// Asks the merge to stop: it fails at its next check (an `InvalidArgument` error;
    /// [`is_cancelled`](Self::is_cancelled) tells it apart) and the index keeps the
    /// segments it was merging.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether [`cancel`](Self::cancel) was called.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub(crate) fn check(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            return Err(crate::TantivyError::InvalidArgument(
                "reordering merge cancelled".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn report(&self, stage: ReorderStage, done: u64, total: u64) {
        if let Some(f) = &self.progress {
            f(stage, done, total);
        }
    }
}

/// A reordering merge request, carried to the merge thread.
pub(crate) struct Reorder {
    pub plan: ReorderPlanFn,
    pub control: ReorderControl,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use super::{ReorderControl, ReorderPlan, ReorderStage};
    use crate::collector::Count;
    use crate::index::Index;
    use crate::indexer::NoMergePolicy;
    use crate::postings::Postings;
    use crate::query::TermQuery;
    use crate::schema::{
        IndexRecordOption, NumericOptions, Schema, TextFieldIndexing, TextOptions, Value, FAST,
        STORED,
    };
    use crate::{
        DocAddress, DocSet, IndexSettings, IndexSortByField, IndexWriter, Order, TantivyDocument,
        Term, TERMINATED,
    };

    /// Two committed segments of docs `0..n` (`id` stored + fast, `rank` = id, the sort
    /// field when `sorted`), text "all w{id % 3}" with positions; doc 4 is deleted.
    fn index(sorted: bool, n: u64) -> Index {
        let mut builder = Schema::builder();
        let text = builder.add_text_field(
            "text",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        let id = builder.add_u64_field(
            "id",
            NumericOptions::default()
                .set_fast()
                .set_stored()
                .set_indexed(),
        );
        let rank = builder.add_u64_field("rank", FAST);
        let flag = builder.add_bool_field("flag", FAST | STORED);
        let settings = IndexSettings {
            sort_by_field: sorted.then(|| IndexSortByField {
                field: "rank".to_string(),
                order: Order::Asc,
            }),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(builder.build())
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let mut writer: IndexWriter = index.writer_for_tests().unwrap();
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..n {
            writer
                .add_document(doc!(
                    text => format!("all w{} all", i % 3),
                    id => i,
                    rank => i,
                    flag => i % 2 == 0,
                ))
                .unwrap();
            if i == n / 2 {
                writer.commit().unwrap();
            }
        }
        writer.delete_term(Term::from_field_u64(id, 4));
        writer.commit().unwrap();
        index
    }

    /// Reverse order by id; `renumber` replaces `rank` with the new doc id.
    fn reverse_plan(renumber: bool) -> super::ReorderPlanFn {
        Box::new(move |readers| {
            let mut docs = Vec::new();
            for (ord, reader) in readers.iter().enumerate() {
                let id = reader.fast_fields().u64("id")?;
                for doc in reader.doc_ids_alive() {
                    docs.push((id.first(doc).unwrap(), DocAddress::new(ord as u32, doc)));
                }
            }
            docs.sort_by(|a, b| b.0.cmp(&a.0));
            let new_doc: std::collections::HashMap<DocAddress, u64> = docs
                .iter()
                .enumerate()
                .map(|(new, d)| (d.1, new as u64))
                .collect();
            Ok(ReorderPlan {
                new_to_old: docs.into_iter().map(|d| d.1).collect(),
                fast_field_values: if renumber {
                    let value: super::FastFieldValueFn = Box::new(move |addr| new_doc[&addr]);
                    vec![("rank".to_string(), value)]
                } else {
                    Vec::new()
                },
            })
        })
    }

    fn merge(
        index: &Index,
        plan: super::ReorderPlanFn,
        control: ReorderControl,
    ) -> crate::Result<()> {
        let mut writer: IndexWriter = index.writer_for_tests().unwrap();
        let ids = index.searchable_segment_ids().unwrap();
        let res = writer.merge_reordered(&ids, plan, control).wait();
        writer.wait_merging_threads().unwrap();
        res.map(|_| ())
    }

    fn check_reversed(index: &Index, n: u64, renumbered: bool) {
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment = searcher.segment_reader(0);
        let schema = index.schema();
        let (text, id_field) = (
            schema.get_field("text").unwrap(),
            schema.get_field("id").unwrap(),
        );
        let alive: Vec<u64> = (0..n).rev().filter(|&i| i != 4).collect();
        assert_eq!(segment.num_docs() as usize, alive.len());
        let id = segment.fast_fields().u64("id").unwrap();
        let rank = segment.fast_fields().u64("rank").unwrap();
        let flag = segment.fast_fields().bool("flag").unwrap();
        let store = segment.get_store_reader(10).unwrap();
        for (doc, &expected) in alive.iter().enumerate() {
            let doc = doc as u32;
            assert_eq!(id.first(doc), Some(expected));
            let r = if renumbered { doc as u64 } else { expected };
            assert_eq!(rank.first(doc), Some(r));
            assert_eq!(flag.first(doc), Some(expected % 2 == 0));
            let stored: TantivyDocument = store.get(doc).unwrap();
            assert_eq!(stored.get_first(id_field).unwrap().as_u64(), Some(expected));
        }
        // postings: every doc of w1 with its positions [1], in new doc order
        let inverted = segment.inverted_index(text).unwrap();
        let term = Term::from_field_text(text, "w1");
        let mut postings = inverted
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            .unwrap()
            .unwrap();
        let mut got = Vec::new();
        let mut positions = Vec::new();
        while postings.doc() != TERMINATED {
            postings.positions(&mut positions);
            assert_eq!(positions, vec![1]);
            got.push(id.first(postings.doc()).unwrap());
            postings.advance();
        }
        let want: Vec<u64> = alive.iter().copied().filter(|i| i % 3 == 1).collect();
        assert_eq!(got, want);
        let all = TermQuery::new(Term::from_field_text(text, "all"), IndexRecordOption::Basic);
        assert_eq!(searcher.search(&all, &Count).unwrap(), alive.len());
    }

    #[test]
    fn reordered_merge_renumbers_the_sort_field() {
        let index = index(true, 5000);
        let progress = Arc::new(AtomicU64::new(0));
        let seen = progress.clone();
        let control = ReorderControl::new().on_progress(move |stage, done, total| {
            assert!(done <= total);
            if stage == ReorderStage::Postings && done == total {
                seen.fetch_add(1, Ordering::Relaxed);
            }
        });
        merge(&index, reverse_plan(true), control).unwrap();
        check_reversed(&index, 5000, true);
        assert_eq!(progress.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reordered_merge_without_sort_field_keeps_fast_values() {
        let index = index(false, 300);
        merge(&index, reverse_plan(false), ReorderControl::new()).unwrap();
        check_reversed(&index, 300, false);
    }

    #[test]
    fn reordered_merge_refuses_an_order_against_the_sort_field() {
        let index = index(true, 300);
        assert!(merge(&index, reverse_plan(false), ReorderControl::new()).is_err());
        assert_eq!(index.searchable_segment_ids().unwrap().len(), 2);
    }

    #[test]
    fn reordered_merge_refuses_a_plan_missing_a_doc() {
        let index = index(false, 300);
        let plan: super::ReorderPlanFn = Box::new(|_| {
            Ok(ReorderPlan {
                new_to_old: vec![DocAddress::new(0, 0)],
                fast_field_values: Vec::new(),
            })
        });
        assert!(merge(&index, plan, ReorderControl::new()).is_err());
        assert_eq!(index.searchable_segment_ids().unwrap().len(), 2);
    }

    #[test]
    fn cancelled_reordered_merge_changes_nothing() {
        let index = index(true, 20_000);
        let before = index.searchable_segment_ids().unwrap();
        let control = ReorderControl::new();
        let cancel = control.clone();
        // cancel from the progress callback, in the middle of the postings
        let control = control.on_progress(move |stage, _, _| {
            if stage == ReorderStage::Postings {
                cancel.cancel();
            }
        });
        let observer = control.clone();
        assert!(merge(&index, reverse_plan(true), control).is_err());
        assert!(observer.is_cancelled());
        assert_eq!(index.searchable_segment_ids().unwrap(), before);
        let reader = index.reader().unwrap();
        assert_eq!(reader.searcher().num_docs(), 19_999);
    }
}
