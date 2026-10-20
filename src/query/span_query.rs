//! Span query: "word A, then within N words word B, then within M words
//! word C…", where every slot is any term source — a plain term, a fuzzy or
//! regex automaton, or any custom [`Automaton`] (e.g. Hebrew
//! prefix/suffix walks).
//!
//! Each slot's matching terms are opened WITH positions and unioned into one
//! positioned posting list ("any word of this slot"), the way
//! `RegexPhraseQuery` does for regexes. Slots are intersected by doc, then
//! each candidate doc's positions are checked against per-gap distances.
//!
//! - Gaps are counted as words BETWEEN two slots: gap 0 = adjacent.
//! - Each gap has its own maximum (`a –2– b –5– c`).
//! - Ordered mode: slots must appear in query order. Unordered mode: each
//!   consecutive pair of slots may appear in either order.
//! - Two slots never match the same word position.
//!
//! The field must be indexed with positions
//! (`IndexRecordOption::WithFreqsAndPositions`). Matching docs get a
//! constant score.

use std::sync::Arc;

use tantivy_fst::Automaton;

use super::phrase_query::RegexPhraseWeight;
use crate::docset::{DocSet, TERMINATED};
use crate::index::SegmentReader;
use crate::postings::{Postings, TermInfo};
use crate::query::explanation::does_not_match;
use crate::query::{
    AutomatonWeight, EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term};
use crate::{DocId, Score};

/// A source of terms for one span slot: anything that can list, for a
/// segment, the term infos it matches.
pub trait SpanTermSource: Send + Sync + 'static {
    /// The term infos this slot matches in `reader`'s segment.
    fn match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>>;
}

impl<A> SpanTermSource for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        self.get_match_term_infos(reader)
    }
}

/// A span slot matching one exact term.
pub struct ExactTermSource {
    term: Term,
}

impl ExactTermSource {
    /// Creates a slot for an exact term.
    pub fn new(term: Term) -> ExactTermSource {
        ExactTermSource { term }
    }
}

impl SpanTermSource for ExactTermSource {
    fn match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.term.field())?;
        Ok(inverted_index
            .get_term_info(&self.term)?
            .into_iter()
            .collect())
    }
}

/// Span query over slots with per-gap maximum word distances.
#[derive(Clone)]
pub struct SpanQuery {
    field: Field,
    slots: Vec<Arc<dyn SpanTermSource>>,
    /// `gaps[i]` = max words between slot i and slot i+1.
    gaps: Vec<u32>,
    ordered: bool,
}

impl std::fmt::Debug for SpanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanQuery")
            .field("field", &self.field)
            .field("slots", &self.slots.len())
            .field("gaps", &self.gaps)
            .field("ordered", &self.ordered)
            .finish()
    }
}

impl SpanQuery {
    /// Starts a span query on `field` with its first slot.
    pub fn new(field: Field, first_slot: Arc<dyn SpanTermSource>) -> SpanQuery {
        SpanQuery {
            field,
            slots: vec![first_slot],
            gaps: Vec::new(),
            ordered: true,
        }
    }

    /// Appends a slot that must occur within `max_words_between` words of
    /// the previous slot.
    pub fn then_within(mut self, max_words_between: u32, slot: Arc<dyn SpanTermSource>) -> Self {
        self.gaps.push(max_words_between);
        self.slots.push(slot);
        self
    }

    /// Ordered (default): slots must appear in query order. Unordered: each
    /// consecutive pair may appear in either order.
    pub fn set_ordered(&mut self, ordered: bool) {
        self.ordered = ordered;
    }

}

impl Query for SpanQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let field_entry = enable_scoring.schema().get_field_entry(self.field);
        let has_positions = field_entry
            .field_type()
            .get_index_record_option()
            .map(IndexRecordOption::has_positions)
            .unwrap_or(false);
        if !has_positions {
            return Err(crate::TantivyError::SchemaError(format!(
                "Applied span query on field {:?}, which does not have positions indexed",
                field_entry.name()
            )));
        }
        Ok(Box::new(SpanWeight {
            field: self.field,
            slots: self.slots.clone(),
            gaps: self.gaps.clone(),
            ordered: self.ordered,
        }))
    }
}

struct SpanWeight {
    field: Field,
    slots: Vec<Arc<dyn SpanTermSource>>,
    gaps: Vec<u32>,
    ordered: bool,
}

type SlotPostings = crate::query::union::SimpleUnion<Box<dyn Postings + 'static>>;

impl SpanWeight {
    fn span_scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Option<SpanScorer>> {
        let inverted_index = reader.inverted_index(self.field)?;
        // Expansion caps are each slot's own business (an AutomatonWeight
        // slot caps by rank); a span never fails for being broad.
        let mut slots = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let term_infos = slot.match_term_infos(reader)?;
            if term_infos.is_empty() {
                return Ok(None); // a slot with no terms can't match
            }
            slots.push(RegexPhraseWeight::get_union_from_term_infos(
                &term_infos,
                reader,
                &inverted_index,
            )?);
        }
        Ok(Some(SpanScorer::new(
            slots,
            self.gaps.clone(),
            self.ordered,
            boost,
        )))
    }
}

impl Weight for SpanWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        match self.span_scorer(reader, boost)? {
            Some(scorer) => Ok(Box::new(scorer)),
            None => Ok(Box::new(EmptyScorer)),
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let Some(mut scorer) = self.span_scorer(reader, 1.0)? else {
            return Err(does_not_match(doc));
        };
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        Ok(Explanation::new("SpanScorer", 1.0))
    }
}

/// Leapfrog intersection of the slot unions + per-doc span check.
struct SpanScorer {
    slots: Vec<SlotPostings>,
    gaps: Vec<u32>,
    ordered: bool,
    boost: Score,
    doc: DocId,
    /// Per-slot positions buffers, reused across docs.
    positions: Vec<Vec<u32>>,
    /// Positions chosen so far by the matcher (for distinctness).
    chosen: Vec<u32>,
}

impl SpanScorer {
    fn new(slots: Vec<SlotPostings>, gaps: Vec<u32>, ordered: bool, boost: Score) -> SpanScorer {
        let num_slots = slots.len();
        let mut scorer = SpanScorer {
            slots,
            gaps,
            ordered,
            boost,
            doc: 0,
            positions: vec![Vec::new(); num_slots],
            chosen: Vec::with_capacity(num_slots),
        };
        scorer.doc = scorer.find_from_current();
        scorer
    }

    /// Aligns all slots on a common doc (starting from slot 0's current doc)
    /// and returns it, or TERMINATED.
    fn align(&mut self) -> DocId {
        let mut candidate = self.slots[0].doc();
        loop {
            if candidate == TERMINATED {
                return TERMINATED;
            }
            let mut all_equal = true;
            for slot in self.slots.iter_mut() {
                let mut doc = slot.doc();
                if doc < candidate {
                    doc = slot.seek(candidate);
                }
                if doc != candidate {
                    candidate = doc; // overshoot: restart alignment from here
                    all_equal = false;
                    break;
                }
            }
            if all_equal {
                return candidate;
            }
        }
    }

    /// From the slots' current state, the next doc that aligns AND passes
    /// the span check.
    fn find_from_current(&mut self) -> DocId {
        loop {
            let doc = self.align();
            if doc == TERMINATED {
                return TERMINATED;
            }
            if self.span_matches() {
                return doc;
            }
            self.slots[0].advance();
        }
    }

    fn span_matches(&mut self) -> bool {
        for (slot, buffer) in self.slots.iter_mut().zip(self.positions.iter_mut()) {
            buffer.clear();
            slot.append_positions_with_offset(0, buffer); // sorted + deduped
        }
        self.chosen.clear();
        let num_first = self.positions[0].len();
        for i in 0..num_first {
            let p = self.positions[0][i];
            self.chosen.push(p);
            if self.extend(1, p) {
                return true;
            }
            self.chosen.pop();
        }
        false
    }

    /// DFS: can slots `slot..` be placed, given slot-1 sits at `prev`?
    fn extend(&mut self, slot: usize, prev: u32) -> bool {
        if slot == self.slots.len() {
            return true;
        }
        let gap = self.gaps[slot - 1];
        // Allowed window for this slot's position q (words between = |q-prev|-1 ≤ gap).
        let lo = if self.ordered {
            prev + 1
        } else {
            prev.saturating_sub(gap + 1)
        };
        let hi = prev.saturating_add(gap + 1);
        let mut idx = self.positions[slot].partition_point(|&q| q < lo);
        while idx < self.positions[slot].len() {
            let q = self.positions[slot][idx];
            if q > hi {
                break;
            }
            idx += 1;
            if q == prev || self.chosen.contains(&q) {
                continue; // one word can't fill two slots
            }
            self.chosen.push(q);
            if self.extend(slot + 1, q) {
                return true;
            }
            self.chosen.pop();
        }
        false
    }
}

impl DocSet for SpanScorer {
    fn advance(&mut self) -> DocId {
        if self.doc == TERMINATED {
            return TERMINATED;
        }
        self.slots[0].advance();
        self.doc = self.find_from_current();
        self.doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc >= target {
            return self.doc;
        }
        self.slots[0].seek(target);
        self.doc = self.find_from_current();
        self.doc
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.slots
            .iter()
            .map(|slot| slot.size_hint())
            .min()
            .unwrap_or(0)
    }
}

impl Scorer for SpanScorer {
    fn score(&mut self) -> Score {
        self.boost
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ExactTermSource, SpanQuery, SpanTermSource};
    use crate::collector::DocSetCollector;
    use crate::query::{AutomatonWeight, FuzzyTermQuery, MatchedTermsSink};
    use crate::schema::{Field, Schema, TEXT};
    use crate::{Index, IndexWriter, Term};

    fn build(lines: &[&str]) -> crate::Result<(Index, Field)> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer: IndexWriter = index.writer_for_tests()?;
        for line in lines {
            writer.add_document(doc!(text => *line))?;
        }
        writer.commit()?;
        Ok((index, text))
    }

    fn exact(field: Field, word: &str) -> Arc<dyn SpanTermSource> {
        Arc::new(ExactTermSource::new(Term::from_field_text(field, word)))
    }

    fn search(index: &Index, query: &SpanQuery) -> crate::Result<Vec<u32>> {
        let searcher = index.reader()?.searcher();
        let mut docs: Vec<u32> = searcher
            .search(query, &DocSetCollector)?
            .into_iter()
            .map(|addr| addr.doc_id)
            .collect();
        docs.sort_unstable();
        Ok(docs)
    }

    #[test]
    fn adjacent_and_gap_counting() -> crate::Result<()> {
        let (index, f) = build(&[
            "a b",       // 0: adjacent
            "a x b",     // 1: 1 word between
            "a x y b",   // 2: 2 words between
            "a x y z b", // 3: 3 words between
        ])?;
        let q = SpanQuery::new(f, exact(f, "a")).then_within(0, exact(f, "b"));
        assert_eq!(search(&index, &q)?, vec![0]);
        let q = SpanQuery::new(f, exact(f, "a")).then_within(2, exact(f, "b"));
        assert_eq!(search(&index, &q)?, vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    fn per_gap_limits() -> crate::Result<()> {
        let (index, f) = build(&[
            "a x b x x x c", // 0: a–1–b–3–c
            "a x x x b c",   // 1: a–3–b–0–c
        ])?;
        // a –2– b –5– c: doc 0 passes (1 ≤ 2, 3 ≤ 5); doc 1 fails the first gap
        let q = SpanQuery::new(f, exact(f, "a"))
            .then_within(2, exact(f, "b"))
            .then_within(5, exact(f, "c"));
        assert_eq!(search(&index, &q)?, vec![0]);
        Ok(())
    }

    #[test]
    fn ordered_vs_unordered() -> crate::Result<()> {
        let (index, f) = build(&["b a", "a b"])?;
        let q = SpanQuery::new(f, exact(f, "a")).then_within(0, exact(f, "b"));
        assert_eq!(search(&index, &q)?, vec![1]);
        let mut q = q;
        q.set_ordered(false);
        assert_eq!(search(&index, &q)?, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn one_word_cannot_fill_two_slots() -> crate::Result<()> {
        let (index, f) = build(&["a b", "a a"])?;
        let mut q = SpanQuery::new(f, exact(f, "a")).then_within(3, exact(f, "a"));
        q.set_ordered(false);
        assert_eq!(search(&index, &q)?, vec![1]); // needs two distinct a's
        Ok(())
    }

    #[test]
    fn automaton_slots_with_sink() -> crate::Result<()> {
        let (index, f) = build(&["japan korea", "japon x korea", "china korea"])?;
        // slot 1: fuzzy "japan"~1 (an AutomatonWeight); slot 2: exact korea
        let fuzzy = FuzzyTermQuery::new(Term::from_field_text(f, "japan"), 1, true);
        let mut weight: AutomatonWeight<_> = AutomatonWeight::new(f, fuzzy.automaton()?);
        let sink = MatchedTermsSink::new();
        weight.set_matched_terms_sink(sink.clone());
        let q = SpanQuery::new(f, Arc::new(weight)).then_within(1, exact(f, "korea"));
        assert_eq!(search(&index, &q)?, vec![0, 1]);
        assert_eq!(sink.take_strings(), vec!["japan".to_string(), "japon".to_string()]);
        Ok(())
    }

    #[test]
    fn missing_slot_matches_nothing() -> crate::Result<()> {
        let (index, f) = build(&["a b"])?;
        let q = SpanQuery::new(f, exact(f, "a")).then_within(5, exact(f, "zzz"));
        assert!(search(&index, &q)?.is_empty());
        Ok(())
    }
}
