use std::collections::BTreeSet;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, OnceLock};

use common::BitSet;
use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::{BitSetDocSet, ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermStreamer, WithoutKeys, ZeroWidthChars};
use crate::{DocId, Score, TantivyError};

/// Collects the terms an [`AutomatonWeight`] matched in the term dictionary
/// while a search runs.
///
/// The automaton-guided walk over the term dictionary already produces the
/// bytes of every matching term; by default they are discarded
/// (`without_keys`). Attach a sink to the query (e.g.
/// [`FuzzyTermQuery::set_matched_terms_sink`](crate::query::FuzzyTermQuery::set_matched_terms_sink))
/// and the same walk stores them here instead — only the matching terms, at
/// the cost of one copy per matched term at scorer-construction time. The
/// per-document scoring loop is unaffected, and queries without a sink run
/// exactly the previous code path.
///
/// Terms are deduplicated across segments (and across repeated scorer
/// constructions, e.g. `explain`). Cloning shares the same underlying store.
#[derive(Clone, Default)]
pub struct MatchedTermsSink {
    matched: Arc<Mutex<BTreeSet<Vec<u8>>>>,
    // Set when an expansion cap left out matching terms (see `is_truncated`).
    truncated: Arc<std::sync::atomic::AtomicBool>,
}

impl MatchedTermsSink {
    /// Creates an empty sink.
    pub fn new() -> MatchedTermsSink {
        MatchedTermsSink::default()
    }

    /// Drains the collected terms as raw term-dictionary bytes
    /// (for `str` fields this is the term's UTF-8 text), sorted and
    /// deduplicated.
    pub fn take_bytes(&self) -> Vec<Vec<u8>> {
        let mut guard = self.lock();
        std::mem::take(&mut *guard).into_iter().collect()
    }

    /// Drains the collected terms as strings. Terms that are not valid UTF-8
    /// are skipped.
    pub fn take_strings(&self) -> Vec<String> {
        let mut guard = self.lock();
        std::mem::take(&mut *guard)
            .into_iter()
            .filter_map(|bytes| String::from_utf8(bytes).ok())
            .collect()
    }

    /// Number of distinct terms collected so far.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Returns true if no term has been collected.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// True when a capped expansion reporting into this sink left out at
    /// least one matching term (in any segment). See
    /// [`AutomatonWeight::set_max_expansions`]. Not reset by `take_*`.
    pub fn is_truncated(&self) -> bool {
        self.truncated.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Records that an expansion feeding this sink was truncated (for
    /// callers that cap an expansion themselves and report it here).
    pub fn mark_truncated(&self) {
        self.truncated
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeSet<Vec<u8>>> {
        self.matched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl fmt::Debug for MatchedTermsSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatchedTermsSink({} terms)", self.len())
    }
}

/// Ranks a matched term for expansion capping: lower is better (e.g. the
/// edit distance of a fuzzy match). See [`AutomatonWeight::set_max_expansions`].
pub type ExpansionRank = Arc<dyn Fn(&[u8]) -> u32 + Send + Sync>;

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
    // When set, the terms matched by the automaton walk are copied into this
    // sink as `scorer()` streams them. `None` keeps the key-less fast path.
    matched_terms_sink: Option<MatchedTermsSink>,
    // When set, at most this many matched terms are used per segment; the
    // rest are dropped by rank (see `matched_terms`). `None` = unlimited.
    max_expansions: Option<usize>,
    // Primary sort key for capping; `None` ranks every term 0.
    expansion_rank: Option<ExpansionRank>,
    // Characters that do not count towards a term's length when capping.
    zero_width: ZeroWidthChars,
    // Terms that always keep a slot under the cap (when they match).
    pinned_terms: Vec<Vec<u8>>,
    // When set, the cap is one cap over all these segments.
    cap_segments: Option<Vec<SegmentReader>>,
    // The expansion over `cap_segments`, computed by the first scorer.
    global_terms: OnceLock<Result<GlobalTerms, String>>,
}

/// The capped expansion over several segments.
struct GlobalTerms {
    terms: Vec<Vec<u8>>,
    truncated: bool,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(field: Field, automaton: IntoArcA) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: None,
            matched_terms_sink: None,
            max_expansions: None,
            expansion_rank: None,
            zero_width: ZeroWidthChars::NONE,
            pinned_terms: Vec::new(),
            cap_segments: None,
            global_terms: OnceLock::new(),
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
            matched_terms_sink: None,
            max_expansions: None,
            expansion_rank: None,
            zero_width: ZeroWidthChars::NONE,
            pinned_terms: Vec::new(),
            cap_segments: None,
            global_terms: OnceLock::new(),
        }
    }

    /// Copies every term the automaton matches into `sink` whenever a scorer
    /// is built (once per segment). See [`MatchedTermsSink`].
    pub fn set_matched_terms_sink(&mut self, sink: MatchedTermsSink) {
        self.matched_terms_sink = Some(sink);
    }

    /// Caps the expansion: when the automaton matches more than `cap` terms
    /// in a segment's dictionary, only the best `cap` of that segment are
    /// used (and reported to the sink). Without a cap, every match is used,
    /// exactly as before.
    ///
    /// Which terms are "best": the lowest [rank](Self::set_expansion_rank)
    /// first, then the shortest (in characters, see
    /// [`set_zero_width_chars`](Self::set_zero_width_chars)), then the
    /// highest document count, then byte order.
    ///
    /// Without a rank (and outside JSON paths) the cap is applied INSIDE the
    /// dictionary walk: the dictionary is walked shortest first, level by
    /// level, and the walk stops once the cap is reached, so the long tail of
    /// a broad pattern is never visited. With a rank the whole expansion is
    /// walked and sorted, since the rank is not known before a term is seen.
    ///
    /// The cap is per segment: an index with several segments may use up to
    /// `cap` terms in each, unless
    /// [`set_cap_across_segments`](Self::set_cap_across_segments) makes it
    /// one cap over all segments.
    ///
    /// When the cap leaves out a matching term, an attached
    /// [`MatchedTermsSink`] is marked truncated
    /// ([`MatchedTermsSink::is_truncated`]). Nothing extra is walked unless
    /// the cap is filled exactly at the end of a length level; then the walk
    /// searches depth first for one more match (stopping at the first; when
    /// none exists it may visit the rest of the matching subtree).
    pub fn set_max_expansions(&mut self, cap: usize) {
        self.max_expansions = Some(cap);
    }

    /// Makes the cap one cap over all of `segments` (normally
    /// [`Searcher::segment_readers`](crate::Searcher::segment_readers)): the
    /// segments are walked together, shortest level first, and a term's
    /// document count is summed over them, so the kept terms do not depend on
    /// how the index is split into segments. Computed once, by the first
    /// scorer. With one segment this is the per-segment cap and costs nothing.
    pub fn set_cap_across_segments(&mut self, segments: &[SegmentReader]) {
        self.cap_segments = (segments.len() > 1).then(|| segments.to_vec());
    }

    /// Terms that always keep a slot under the cap when the automaton matches
    /// them and the dictionary has them (e.g. the exact word typed), even when
    /// they would not rank among the best `cap`; the worst kept slot makes
    /// room.
    pub fn set_pinned_terms(&mut self, terms: Vec<Vec<u8>>) {
        self.pinned_terms = terms;
    }

    /// The primary key the cap sorts by (e.g. a fuzzy match's edit distance).
    pub fn set_expansion_rank(&mut self, rank: ExpansionRank) {
        self.expansion_rank = Some(rank);
    }

    /// Characters that do not count towards a term's length when the cap
    /// prefers shorter terms, e.g. an intra-word apostrophe, quote or
    /// combining mark the tokenizer keeps. Under a cap, terms that differ only
    /// in them are one slot. See [`ZeroWidthChars`].
    pub fn set_zero_width_chars(&mut self, zero_width: ZeroWidthChars) {
        self.zero_width = zero_width;
    }

    fn uses_shortest_first_walk(&self) -> bool {
        self.expansion_rank.is_none() && self.json_path_bytes.is_none()
    }

    /// Level-by-level walk of the dictionaries in lockstep (shortest level of
    /// any dictionary first); stops at `cap` slots. A slot is one
    /// zero-width-stripped form; its document count is summed over its terms
    /// and dictionaries. Ties on the last level go to the higher document
    /// count, then byte order. Also returns whether the cap left out a
    /// matching term.
    fn cap_shortest_first(
        &self,
        dicts: &[&TermDictionary],
        cap: usize,
        check_more: bool,
    ) -> io::Result<(Vec<StrippedGroup>, bool)> {
        let automaton: &A = &self.automaton;
        let mut walks: Vec<_> = dicts
            .iter()
            .map(|dict| dict.search_shortest_first(automaton, self.zero_width))
            .collect();
        let mut pending = Vec::with_capacity(walks.len());
        for walk in walks.iter_mut() {
            pending.push(walk.next_level()?);
        }
        let mut kept: Vec<StrippedGroup> = Vec::new();
        let mut truncated = false;
        while kept.len() < cap {
            let Some(length) = pending.iter().flatten().map(|(len, _)| *len).min() else {
                break;
            };
            let mut level = Vec::new();
            for (slot, walk) in pending.iter_mut().zip(walks.iter_mut()) {
                if slot.as_ref().is_some_and(|(len, _)| *len == length) {
                    let (_, terms) = slot.take().expect("checked above");
                    level.extend(terms);
                    *slot = walk.next_level()?;
                }
            }
            let mut groups = group_by_stripped(level, self.zero_width);
            let room = cap - kept.len();
            if groups.len() > room {
                truncated = true;
                groups.sort_unstable_by(|a, b| {
                    b.doc_freq
                        .cmp(&a.doc_freq)
                        .then_with(|| a.stripped.cmp(&b.stripped))
                });
                groups.truncate(room);
            }
            kept.extend(groups);
        }
        if !truncated && kept.len() >= cap && check_more {
            truncated = pending.iter().any(Option::is_some);
            for walk in walks.iter_mut() {
                truncated = truncated || walk.has_more()?;
            }
        }
        Ok((kept, truncated))
    }

    /// Full walk of the dictionaries, then sort the slots (one per
    /// zero-width-stripped form) by (rank, shortest, most frequent, bytes)
    /// and keep `cap`. A slot's rank is its best member's, its document count
    /// the sum over its terms and dictionaries. Also returns whether the cap
    /// left out a matching term.
    fn cap_full_walk(
        &self,
        dicts: &[&TermDictionary],
        cap: usize,
    ) -> io::Result<(Vec<StrippedGroup>, bool)> {
        let mut by_form: std::collections::HashMap<Vec<u8>, (u32, usize, StrippedGroup)> =
            std::collections::HashMap::new();
        for dict in dicts {
            let mut stream = self.automaton_stream_with_keys(dict)?;
            while stream.advance() {
                let key = stream.key();
                let rank = self.expansion_rank.as_ref().map(|r| r(key)).unwrap_or(0);
                let chars = self.zero_width.term_len(key);
                let form = self.zero_width.strip(key).into_owned();
                let slot = by_form.entry(form.clone()).or_insert((
                    u32::MAX,
                    chars,
                    StrippedGroup {
                        stripped: form,
                        doc_freq: 0,
                        terms: Vec::new(),
                    },
                ));
                slot.0 = slot.0.min(rank);
                slot.2.doc_freq += u64::from(stream.value().doc_freq);
                if !slot.2.terms.iter().any(|t| t.as_slice() == key) {
                    slot.2.terms.push(key.to_vec());
                }
            }
        }
        let mut slots: Vec<(u32, usize, StrippedGroup)> = by_form.into_values().collect();
        slots.sort_unstable_by(|a, b| {
            (a.0, a.1, std::cmp::Reverse(a.2.doc_freq), &a.2.stripped).cmp(&(
                b.0,
                b.1,
                std::cmp::Reverse(b.2.doc_freq),
                &b.2.stripped,
            ))
        });
        let truncated = slots.len() > cap;
        slots.truncate(cap);
        Ok((slots.into_iter().map(|s| s.2).collect(), truncated))
    }

    /// The kept terms of a capped expansion over `dicts` (several = one cap
    /// for all of them), pinned terms included, and whether the cap left out
    /// a matching term.
    fn capped_terms(
        &self,
        dicts: &[&TermDictionary],
        cap: usize,
    ) -> io::Result<(Vec<Vec<u8>>, bool)> {
        let (mut slots, mut truncated) = if self.uses_shortest_first_walk() {
            let check_more = self.matched_terms_sink.is_some() || dicts.len() > 1;
            self.cap_shortest_first(dicts, cap, check_more)?
        } else {
            self.cap_full_walk(dicts, cap)?
        };
        // Pinned terms always keep a slot (taken from the worst kept slot).
        for pin in &self.pinned_terms {
            let form = self.zero_width.strip(pin);
            if slots.iter().any(|s| s.stripped.as_slice() == &*form) || !self.accepts(pin) {
                continue;
            }
            let mut present = false;
            for dict in dicts {
                present = present || dict.get(pin)?.is_some();
            }
            if !present {
                continue;
            }
            if cap == 0 {
                break;
            }
            if slots.len() >= cap {
                slots.truncate(cap - 1);
                truncated = true;
            }
            slots.insert(
                0,
                StrippedGroup {
                    stripped: form.into_owned(),
                    doc_freq: 0,
                    terms: vec![pin.clone()],
                },
            );
        }
        Ok((slots.into_iter().flat_map(|s| s.terms).collect(), truncated))
    }

    /// Whether the automaton matches `term`.
    fn accepts(&self, term: &[u8]) -> bool {
        let automaton: &A = &self.automaton;
        let mut state = automaton.start();
        for &byte in term {
            if !automaton.can_match(&state) {
                return false;
            }
            state = automaton.accept(&state, byte);
        }
        automaton.is_match(&state)
    }

    /// The capped expansion over every segment set by
    /// [`set_cap_across_segments`](Self::set_cap_across_segments), computed
    /// once.
    fn global_terms(&self, segments: &[SegmentReader], cap: usize) -> io::Result<&GlobalTerms> {
        let computed = self.global_terms.get_or_init(|| {
            let mut indexes = Vec::with_capacity(segments.len());
            for reader in segments {
                indexes.push(
                    reader
                        .inverted_index(self.field)
                        .map_err(|err| err.to_string())?,
                );
            }
            let dicts: Vec<&TermDictionary> = indexes.iter().map(|i| i.terms()).collect();
            let (terms, truncated) = self
                .capped_terms(&dicts, cap)
                .map_err(|err| err.to_string())?;
            Ok(GlobalTerms { terms, truncated })
        });
        computed
            .as_ref()
            .map_err(|err| io::Error::other(err.clone()))
    }

    /// Every term the automaton matches in `term_dict`, capped and reported
    /// to the sink as configured.
    fn matched_terms(&self, term_dict: &TermDictionary) -> io::Result<Vec<TermInfo>> {
        if let Some(cap) = self.max_expansions {
            let local;
            let (kept, truncated): (&[Vec<u8>], bool) = match &self.cap_segments {
                Some(segments) => {
                    let global = self.global_terms(segments, cap)?;
                    (&global.terms, global.truncated)
                }
                None => {
                    local = self.capped_terms(&[term_dict], cap)?;
                    (&local.0, local.1)
                }
            };
            if truncated {
                if let Some(sink) = &self.matched_terms_sink {
                    sink.mark_truncated();
                }
            }
            let mut infos = Vec::with_capacity(kept.len());
            let mut sink = self.matched_terms_sink.as_ref().map(|s| s.lock());
            for key in kept {
                if let Some(info) = term_dict.get(key)? {
                    infos.push(info);
                    if let Some(sink) = sink.as_mut() {
                        sink.insert(key.clone());
                    }
                }
            }
            return Ok(infos);
        }
        // No cap: the plain stream (keys only when a sink wants them).
        if let Some(sink) = &self.matched_terms_sink {
            let mut term_stream = self.automaton_stream_with_keys(term_dict)?;
            let mut infos = Vec::new();
            let mut guard = sink.lock();
            while term_stream.advance() {
                guard.insert(term_stream.key().to_vec());
                infos.push(term_stream.value().clone());
            }
            return Ok(infos);
        }
        let mut term_stream = self.automaton_stream(term_dict)?;
        let mut infos = Vec::new();
        while term_stream.advance() {
            infos.push(term_stream.value().clone());
        }
        Ok(infos)
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamer<'a, &'a A, WithoutKeys>> {
        let automaton: &A = &self.automaton;
        // Matching terms are only used for their `TermInfo`.
        let mut term_stream_builder = term_dict.search(automaton).without_keys();

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    /// Like [`automaton_stream`](Self::automaton_stream), but keeps the term
    /// bytes so they can be reported through a [`MatchedTermsSink`].
    fn automaton_stream_with_keys<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamer<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    /// Returns the term infos that match the automaton
    ///
    /// Honors the expansion cap; if a [`MatchedTermsSink`] is attached, the
    /// (kept) matched terms are also copied into it.
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.field)?;
        Ok(self.matched_terms(inverted_index.terms())?)
    }

    /// The field this weight searches.
    pub fn field(&self) -> Field {
        self.field
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut doc_bitset = BitSet::with_max_value(max_doc);
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        for term_info in self.matched_terms(term_dict)? {
            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
            loop {
                let docs = block_segment_postings.docs();
                if docs.is_empty() {
                    break;
                }
                for &doc in docs {
                    doc_bitset.insert(doc);
                }
                block_segment_postings.advance();
            }
        }
        let doc_bitset = BitSetDocSet::from(doc_bitset);
        let const_scorer = ConstScorer::new(doc_bitset, boost);
        Ok(Box::new(const_scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            Ok(Explanation::new("AutomatonScorer", 1.0))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

/// The terms of one shortest-first level that share a zero-width-stripped
/// form: one expansion slot under a cap.
struct StrippedGroup {
    stripped: Vec<u8>,
    doc_freq: u64,
    terms: Vec<Vec<u8>>,
}

/// Groups a level's terms by their zero-width-stripped form, keeping the
/// level's byte order of first appearance.
fn group_by_stripped(
    level: Vec<(Vec<u8>, TermInfo)>,
    zero_width: crate::termdict::ZeroWidthChars,
) -> Vec<StrippedGroup> {
    let mut index: std::collections::HashMap<Vec<u8>, usize> = std::collections::HashMap::new();
    let mut groups: Vec<StrippedGroup> = Vec::new();
    for (key, info) in level {
        let stripped = zero_width.strip(&key).into_owned();
        let at = *index.entry(stripped.clone()).or_insert_with(|| {
            groups.push(StrippedGroup {
                stripped,
                doc_freq: 0,
                terms: Vec::new(),
            });
            groups.len() - 1
        });
        groups[at].doc_freq += u64::from(info.doc_freq);
        // The same term from several segments is one term.
        if !groups[at].terms.contains(&key) {
            groups[at].terms.push(key);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_matched_terms_sink() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let mut automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        // Same matches as without the sink...
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.advance(), TERMINATED);
        // ...plus the matched words, and only the matched words.
        assert_eq!(
            sink.take_strings(),
            vec!["abc".to_string(), "abcd".to_string()]
        );
        assert!(sink.is_empty(), "take_strings drains the sink");
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_keeps_shortest() -> crate::Result<()> {
        // "abc" and "abcd" both match; cap 1 keeps the shorter "abc" only
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let mut automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        automaton_weight.set_max_expansions(1);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32); // "abc"
        assert_eq!(scorer.advance(), TERMINATED); // "abcd" (doc 2) dropped
        assert_eq!(sink.take_strings(), vec!["abc".to_string()]);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_reports_truncation() -> crate::Result<()> {
        // "abc" and "abcd" match: cap 1 truncates, cap 2 (filled exactly) and
        // cap 3 do not
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        for (cap, truncated) in [(1, true), (2, false), (3, false)] {
            let mut automaton_weight = AutomatonWeight::new(field, PrefixedByA);
            let sink = super::MatchedTermsSink::new();
            automaton_weight.set_matched_terms_sink(sink.clone());
            automaton_weight.set_max_expansions(cap);
            automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert_eq!(sink.is_truncated(), truncated, "cap {cap}");
        }
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_zero_width_chars() -> crate::Result<()> {
        // "abc" (3 chars) and "a\"bc" (4 chars, 3 with a zero-width quote) both
        // match; cap 1 keeps the shorter. With the quote zero-width they are
        // the same form ("abc"), so they share the one slot and both are kept
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"a\"bc"))?;
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut automaton_weight = AutomatonWeight::new(title, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        automaton_weight.set_max_expansions(1);
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(sink.take_strings(), vec!["abc".to_string()]);

        automaton_weight.set_zero_width_chars(crate::termdict::ZeroWidthChars::new(b"\""));
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.advance(), 1u32);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(
            sink.take_strings(),
            vec!["a\"bc".to_string(), "abc".to_string()]
        );
        Ok(())
    }

    #[test]
    fn test_automaton_weight_zero_width_variants_share_one_slot() -> crate::Result<()> {
        // Level of 3-char forms: "abc" (+ its variant "a\"bc"), "abd", "abe".
        // With cap 2, the variants count as ONE slot: the cut keeps "abc"
        // (with "a\"bc") and the more frequent of "abd" / "abe", instead of
        // spending a second slot on the quoted duplicate. Their summed doc
        // count ranks the "abc" form first.
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        for t in ["abc", "a\"bc", "abd", "abd", "abe"] {
            index_writer.add_document(doc!(title=>t))?;
        }
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut automaton_weight = AutomatonWeight::new(title, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        automaton_weight.set_max_expansions(2);
        automaton_weight.set_zero_width_chars(crate::termdict::ZeroWidthChars::new(b"\""));
        let _ = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        let mut kept = sink.take_strings();
        kept.sort();
        assert_eq!(
            kept,
            vec!["a\"bc".to_string(), "abc".to_string(), "abd".to_string()]
        );

        // the full-walk path (a rank is set) groups the same way
        automaton_weight.set_expansion_rank(std::sync::Arc::new(|_key: &[u8]| 0));
        let _ = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        let mut kept = sink.take_strings();
        kept.sort();
        assert_eq!(
            kept,
            vec!["a\"bc".to_string(), "abc".to_string(), "abd".to_string()]
        );
        Ok(())
    }

    #[test]
    fn test_automaton_weight_multibyte_zero_width_variants_share_one_slot() -> crate::Result<()> {
        // As above with a multi-byte zero-width character (a combining acute
        // accent): "ab\u{301}c" is as short as "abc" and shares its slot.
        static COMBINING: [(char, char); 1] = [('\u{300}', '\u{36F}')];
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        for t in ["abc", "ab\u{301}c", "abd", "abd", "abe"] {
            index_writer.add_document(doc!(title=>t))?;
        }
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut automaton_weight = AutomatonWeight::new(title, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        automaton_weight.set_max_expansions(2);
        let expected = vec![
            "abc".to_string(),
            "abd".to_string(),
            "ab\u{301}c".to_string(),
        ];
        automaton_weight
            .set_zero_width_chars(crate::termdict::ZeroWidthChars::NONE.with_ranges(&COMBINING));
        let _ = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        let mut kept = sink.take_strings();
        kept.sort();
        assert_eq!(kept, expected);

        // the full-walk path (a rank is set) groups the same way
        automaton_weight.set_expansion_rank(std::sync::Arc::new(|_key: &[u8]| 0));
        let _ = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        let mut kept = sink.take_strings();
        kept.sort();
        assert_eq!(kept, expected);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_rank_first() -> crate::Result<()> {
        // a rank that prefers "abcd" overrides the shortest-first tie rule
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let mut automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        automaton_weight.set_matched_terms_sink(sink.clone());
        automaton_weight.set_max_expansions(1);
        automaton_weight.set_expansion_rank(std::sync::Arc::new(|key: &[u8]| {
            if key == b"abcd" {
                0
            } else {
                1
            }
        }));
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(sink.take_strings(), vec!["abcd".to_string()]);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }

    /// Terms the capped walk keeps on `index`, with one cap over all its
    /// segments, and whether it reports truncation.
    fn kept_across_segments(index: &Index, cap: usize, pins: &[&str]) -> (Vec<String>, bool) {
        let field = index.schema().get_field("title").unwrap();
        let searcher = index.reader().unwrap().searcher();
        let mut weight = AutomatonWeight::new(field, PrefixedByA);
        let sink = super::MatchedTermsSink::new();
        weight.set_matched_terms_sink(sink.clone());
        weight.set_max_expansions(cap);
        weight.set_pinned_terms(pins.iter().map(|p| p.as_bytes().to_vec()).collect());
        weight.set_cap_across_segments(searcher.segment_readers());
        for reader in searcher.segment_readers() {
            weight.scorer(reader, 1.0).unwrap();
        }
        (sink.take_strings(), sink.is_truncated())
    }

    fn index_in_segments(segments: &[&[&str]]) -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(crate::indexer::NoMergePolicy));
        for docs in segments {
            for doc in *docs {
                index_writer.add_document(doc!(title=>*doc))?;
            }
            index_writer.commit()?;
        }
        Ok(index)
    }

    #[test]
    fn test_automaton_weight_cap_across_segments_ignores_layout() -> crate::Result<()> {
        // "ab" and "ax" tie on length; "ax" is more frequent once the segments
        // are summed, though "ab" wins in the first segment alone.
        let docs: [&str; 7] = ["ab", "ab", "ax", "abc", "ax", "ax", "abcd"];
        let one = index_in_segments(&[&docs])?;
        let three = index_in_segments(&[&docs[..3], &docs[3..5], &docs[5..]])?;
        assert_eq!(three.searchable_segment_ids()?.len(), 3);
        for cap in [1, 2, 3, 4] {
            let expected = kept_across_segments(&one, cap, &[]);
            assert_eq!(
                kept_across_segments(&three, cap, &[]),
                expected,
                "cap {cap}"
            );
        }
        assert_eq!(
            kept_across_segments(&three, 1, &[]),
            (vec!["ax".to_string()], true)
        );
        assert_eq!(kept_across_segments(&three, 4, &[]).1, false);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_pinned_term_keeps_a_slot() -> crate::Result<()> {
        let docs: [&str; 4] = ["ab", "ax", "abcd", "bcd"];
        let index = index_in_segments(&[&docs[..2], &docs[2..]])?;
        // Cap 2 keeps "ab" and "ax"; the pinned "abcd" takes the worst slot.
        let (kept, truncated) = kept_across_segments(&index, 2, &["abcd"]);
        assert_eq!(kept, vec!["ab".to_string(), "abcd".to_string()]);
        assert!(truncated);
        // A pin the automaton does not match, or the index lacks, is ignored.
        let (kept, _) = kept_across_segments(&index, 2, &["bcd", "azz"]);
        assert_eq!(kept, vec!["ab".to_string(), "ax".to_string()]);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_cap_across_segments_term_once_per_segment() -> crate::Result<()> {
        // "ab" in both segments: each segment reads its postings once.
        let index = index_in_segments(&[&["ab", "abc"], &["ab"]])?;
        let field = index.schema().get_field("title").unwrap();
        let searcher = index.reader()?.searcher();
        let mut weight = AutomatonWeight::new(field, PrefixedByA);
        weight.set_max_expansions(5);
        weight.set_cap_across_segments(searcher.segment_readers());
        let counts: Vec<usize> = searcher
            .segment_readers()
            .iter()
            .map(|r| weight.get_match_term_infos(r).unwrap().len())
            .collect();
        assert_eq!(counts, vec![2, 1]);
        Ok(())
    }
}
