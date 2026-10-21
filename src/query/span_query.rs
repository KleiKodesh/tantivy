//! Span query: "word A, then within N words word B, then within M words
//! word C…", where every slot is any term source — a plain term, a fuzzy or
//! regex automaton, or any custom [`Automaton`] (e.g. a language-specific
//! prefix/suffix walk).
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
//! - Window mode ([`SpanQuery::set_window`]): the per-gap limits are replaced
//!   by one window: every slot at its own position, in any order, the last
//!   position at most `max_span` after the first. Checked in one sweep over
//!   the slots' merged positions (a bipartite slot/position matching runs
//!   only when one position can fill several slots).
//!
//! Ordered mode is checked with one forward sweep per slot over the sorted
//! positions (the positions where a valid chain over slots `0..=i` can end),
//! so the check is linear in the positions, with no backtracking.
//!
//! The field must be indexed with positions
//! (`IndexRecordOption::WithFreqsAndPositions`). Matching docs get a
//! constant score.

use std::collections::VecDeque;
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
    /// Window mode: `Some(max_span)` (see [`SpanQuery::set_window`]).
    window: Option<u32>,
}

impl std::fmt::Debug for SpanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanQuery")
            .field("field", &self.field)
            .field("slots", &self.slots.len())
            .field("gaps", &self.gaps)
            .field("ordered", &self.ordered)
            .field("window", &self.window)
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
            window: None,
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

    /// Window mode: every slot at its own word position, in any order, all
    /// of them inside one window whose last position is at most `max_span`
    /// after its first (`max_span` = 0 fits one word only). Replaces the
    /// per-gap limits and the order flag. `None` restores them.
    pub fn set_window(&mut self, max_span: Option<u32>) {
        self.window = max_span;
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
            window: self.window,
        }))
    }
}

struct SpanWeight {
    field: Field,
    slots: Vec<Arc<dyn SpanTermSource>>,
    gaps: Vec<u32>,
    ordered: bool,
    window: Option<u32>,
}

type SlotPostings = crate::query::union::SimpleUnion<Box<dyn Postings + 'static>>;

impl SpanWeight {
    fn span_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<SpanScorer>> {
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
            self.window,
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
    window: Option<u32>,
    boost: Score,
    doc: DocId,
    /// Per-slot positions buffers, reused across docs.
    positions: Vec<Vec<u32>>,
    /// Positions chosen so far by the matcher (for distinctness).
    chosen: Vec<u32>,
    /// Ordered sweep: chain ends so far / next.
    feasible: Vec<u32>,
    next: Vec<u32>,
    /// Window mode scratch, reused across docs.
    windows: WindowMatcher,
}

impl SpanScorer {
    fn new(
        slots: Vec<SlotPostings>,
        gaps: Vec<u32>,
        ordered: bool,
        window: Option<u32>,
        boost: Score,
    ) -> SpanScorer {
        let num_slots = slots.len();
        let mut scorer = SpanScorer {
            slots,
            gaps,
            ordered,
            window,
            boost,
            doc: 0,
            positions: vec![Vec::new(); num_slots],
            chosen: Vec::with_capacity(num_slots),
            feasible: Vec::new(),
            next: Vec::new(),
            windows: WindowMatcher::default(),
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
        if let Some(max_span) = self.window {
            return self.windows.matches(&self.positions, max_span);
        }
        if self.ordered {
            return self.ordered_matches();
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

    /// Ordered mode: positions `p0 < p1 < ...` (one per slot) with
    /// `p[i+1] - p[i] - 1 <= gaps[i]`. A forward sweep keeps every position
    /// where a valid chain over the slots so far can end; each step is a
    /// linear two-pointer merge of two sorted lists.
    fn ordered_matches(&mut self) -> bool {
        self.feasible.clear();
        self.feasible.extend_from_slice(&self.positions[0]);
        for slot in 1..self.positions.len() {
            let window = self.gaps[slot - 1] as u64 + 1;
            self.next.clear();
            let mut j = 0;
            for &q in &self.positions[slot] {
                let lo = (q as u64).saturating_sub(window);
                while j < self.feasible.len() && (self.feasible[j] as u64) < lo {
                    j += 1;
                }
                if j < self.feasible.len() && self.feasible[j] < q {
                    self.next.push(q);
                }
            }
            std::mem::swap(&mut self.feasible, &mut self.next);
            if self.feasible.is_empty() {
                return false;
            }
        }
        !self.feasible.is_empty()
    }

    /// DFS (unordered pairwise mode): can slots `slot..` be placed, given
    /// slot-1 sits at `prev`?
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

const UNMATCHED: usize = usize::MAX;

/// Window mode's check, with its scratch reused across docs.
///
/// One sweep over the slots' occurrences in position order. The window
/// keeps, per slot, the positions it can take, and a slot/position matching
/// that is kept up to date incrementally: a position leaving the window
/// frees its slot, and only free slots are augmented (Kuhn), so a matching
/// is never rebuilt from scratch. Positions are numbered once per doc.
#[derive(Default)]
struct WindowMatcher {
    /// (position, slot) of every slot occurrence, sorted.
    entries: Vec<(u32, usize)>,
    /// Position number (0.. over the distinct positions) of each entry.
    position_ids: Vec<usize>,
    /// Occurrences per slot inside the window.
    coverage: Vec<u32>,
    /// Per slot: the numbers of the positions it can take in the window.
    in_window: Vec<VecDeque<usize>>,
    /// Per position number: the slot it is matched to, or `UNMATCHED`.
    owner: Vec<usize>,
    /// Per slot: its matched position number, or `UNMATCHED`.
    taken: Vec<usize>,
    /// Visit marks of the current augmenting search (`generation`).
    stamp: Vec<u32>,
    generation: u32,
    /// Augmenting path: (slot, next candidate index in its `in_window`).
    stack: Vec<(usize, usize)>,
}

impl WindowMatcher {
    /// True when every slot gets its own position inside one window of at
    /// most `max_span`. `positions[s]` = slot `s`'s sorted positions.
    fn matches(&mut self, positions: &[Vec<u32>], max_span: u32) -> bool {
        let k = positions.len();
        self.entries.clear();
        for (slot, list) in positions.iter().enumerate() {
            self.entries.extend(list.iter().map(|&p| (p, slot)));
        }
        self.entries.sort_unstable();
        self.position_ids.clear();
        let mut next_id = 0;
        for i in 0..self.entries.len() {
            if i > 0 && self.entries[i - 1].0 != self.entries[i].0 {
                next_id += 1;
            }
            self.position_ids.push(next_id);
        }
        let position_count = next_id + 1;
        self.coverage.clear();
        self.coverage.resize(k, 0);
        self.in_window.resize_with(k, VecDeque::new);
        self.in_window.iter_mut().for_each(VecDeque::clear);
        self.owner.clear();
        self.owner.resize(position_count, UNMATCHED);
        self.taken.clear();
        self.taken.resize(k, UNMATCHED);
        self.stamp.clear();
        self.stamp.resize(position_count, 0);
        self.generation = 0;

        let len = self.entries.len();
        let (mut covered, mut distinct, mut matched) = (0usize, 0usize, 0usize);
        let mut left = 0;
        for right in 0..len {
            let (position, slot) = self.entries[right];
            let id = self.position_ids[right];
            covered += usize::from(self.coverage[slot] == 0);
            self.coverage[slot] += 1;
            self.in_window[slot].push_back(id);
            if right == 0 || self.position_ids[right - 1] != id {
                distinct += 1;
            }
            while position - self.entries[left].0 > max_span {
                let gone = self.entries[left].1;
                let gone_id = self.position_ids[left];
                self.coverage[gone] -= 1;
                covered -= usize::from(self.coverage[gone] == 0);
                self.in_window[gone].pop_front();
                if self.position_ids[left + 1] != gone_id {
                    distinct -= 1; // the position's last entry left
                }
                let holder = self.owner[gone_id];
                if holder != UNMATCHED {
                    self.owner[gone_id] = UNMATCHED;
                    self.taken[holder] = UNMATCHED;
                    matched -= 1;
                }
                left += 1;
            }
            // Every entry of this position must be in before deciding.
            if right + 1 < len && self.position_ids[right + 1] == id {
                continue;
            }
            if covered < k || distinct < k {
                continue;
            }
            for s in 0..k {
                if self.taken[s] == UNMATCHED {
                    if !self.augment(s) {
                        break;
                    }
                    matched += 1;
                }
            }
            if matched == k {
                return true;
            }
        }
        false
    }

    /// Kuhn's augmenting path from the free slot `start`, iterative; on
    /// success the path is flipped and `start` is matched.
    fn augment(&mut self, start: usize) -> bool {
        self.generation += 1;
        let generation = self.generation;
        self.stack.clear();
        self.stack.push((start, 0));
        while let Some(&(slot, i)) = self.stack.last() {
            let Some(&id) = self.in_window[slot].get(i) else {
                self.stack.pop();
                continue;
            };
            self.stack.last_mut().expect("non-empty").1 = i + 1;
            if self.stamp[id] == generation {
                continue;
            }
            self.stamp[id] = generation;
            let holder = self.owner[id];
            if holder == UNMATCHED {
                // Each frame takes the position it tried last.
                for &(s, next) in &self.stack {
                    let taken = self.in_window[s][next - 1];
                    self.owner[taken] = s;
                    self.taken[s] = taken;
                }
                return true;
            }
            self.stack.push((holder, 0));
        }
        false
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
        assert_eq!(
            sink.take_strings(),
            vec!["japan".to_string(), "japon".to_string()]
        );
        Ok(())
    }

    #[test]
    fn window_mode_any_order_distinct_positions() -> crate::Result<()> {
        let (index, f) = build(&[
            "a b",         // 0
            "b a",         // 1
            "b x x a",     // 2: span 3
            "c x a b",     // 3: a, b, c span 3
            "a x a",       // 4: two a's, span 2
            "a",           // 5: one a
            "b a x x x c", // 6: span 5
        ])?;
        let window = |words: &[&str], span: u32| -> crate::Result<Vec<u32>> {
            let mut q = SpanQuery::new(f, exact(f, words[0]));
            for w in &words[1..] {
                q = q.then_within(0, exact(f, w));
            }
            q.set_window(Some(span));
            search(&index, &q)
        };
        assert_eq!(window(&["a", "b"], 1)?, vec![0, 1, 3, 6]);
        assert_eq!(window(&["b", "a"], 3)?, vec![0, 1, 2, 3, 6]);
        assert_eq!(window(&["a", "b", "c"], 3)?, vec![3]);
        assert_eq!(window(&["a", "b", "c"], 5)?, vec![3, 6]);
        assert_eq!(window(&["a", "a"], 1)?, Vec::<u32>::new());
        assert_eq!(window(&["a", "a"], 2)?, vec![4]);
        Ok(())
    }

    #[test]
    fn window_mode_shared_positions_need_a_matching() -> crate::Result<()> {
        // slot 0 = any one-letter word ("a"~1), slot 1 = "a".
        let (index, f) = build(&["a b", "a x", "a a", "b b"])?;
        let any_letter = FuzzyTermQuery::new(Term::from_field_text(f, "a"), 1, true);
        let weight: AutomatonWeight<_> = AutomatonWeight::new(f, any_letter.automaton()?);
        let mut q = SpanQuery::new(f, Arc::new(weight)).then_within(0, exact(f, "a"));
        q.set_window(Some(1));
        assert_eq!(search(&index, &q)?, vec![0, 1, 2]);
        let mut q = SpanQuery::new(f, exact(f, "a")).then_within(0, exact(f, "a"));
        q.set_window(Some(1));
        assert_eq!(search(&index, &q)?, vec![2]);
        Ok(())
    }

    #[test]
    fn ordered_sweep_takes_a_later_start() -> crate::Result<()> {
        // the first "a" is too far from "b"; the second one reaches it
        let (index, f) = build(&["a x x a b c", "a x x x b"])?;
        let q = SpanQuery::new(f, exact(f, "a"))
            .then_within(0, exact(f, "b"))
            .then_within(0, exact(f, "c"));
        assert_eq!(search(&index, &q)?, vec![0]);
        Ok(())
    }

    /// Every injective slot -> position assignment, brute force.
    fn window_reference(positions: &[Vec<u32>], max_span: u32) -> bool {
        fn place(
            slot: usize,
            positions: &[Vec<u32>],
            chosen: &mut Vec<u32>,
            max_span: u32,
        ) -> bool {
            if slot == positions.len() {
                let lo = chosen.iter().min().copied().unwrap_or(0);
                let hi = chosen.iter().max().copied().unwrap_or(0);
                return hi - lo <= max_span;
            }
            for &p in &positions[slot] {
                if chosen.contains(&p) {
                    continue;
                }
                chosen.push(p);
                if place(slot + 1, positions, chosen, max_span) {
                    return true;
                }
                chosen.pop();
            }
            false
        }
        place(0, positions, &mut Vec::new(), max_span)
    }

    #[test]
    fn window_matcher_equals_brute_force() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let mut matcher = super::WindowMatcher::default();
        for _ in 0..3000 {
            let k = 1 + next(4) as usize;
            let len = 1 + next(10) as u32;
            // each slot takes a random subset of the positions (shared ones too)
            let positions: Vec<Vec<u32>> = (0..k)
                .map(|_| (0..len).filter(|_| next(3) == 0).collect())
                .collect();
            if positions.iter().any(Vec::is_empty) {
                continue;
            }
            let max_span = next(6) as u32;
            assert_eq!(
                matcher.matches(&positions, max_span),
                window_reference(&positions, max_span),
                "{positions:?} span {max_span}"
            );
        }
    }

    #[test]
    fn window_matcher_dense_broad_slot_is_fast() {
        // 2,000 positions; three slots take every position (a broad walk),
        // two take every sixth: no window of span 4 holds two of those, so
        // every window gets matched and fails, with augmenting paths through
        // the broad slots.
        let all: Vec<u32> = (0..2000).collect();
        let sparse: Vec<u32> = (0..2000).step_by(6).collect();
        let positions = vec![all.clone(), all.clone(), all, sparse.clone(), sparse];
        let mut matcher = super::WindowMatcher::default();
        let t = std::time::Instant::now();
        assert!(!matcher.matches(&positions, 4));
        assert!(matcher.matches(&positions, 6));
        assert!(
            t.elapsed() < std::time::Duration::from_secs(2),
            "{:?}",
            t.elapsed()
        );
    }

    #[test]
    fn missing_slot_matches_nothing() -> crate::Result<()> {
        let (index, f) = build(&["a b"])?;
        let q = SpanQuery::new(f, exact(f, "a")).then_within(5, exact(f, "zzz"));
        assert!(search(&index, &q)?.is_empty());
        Ok(())
    }
}
