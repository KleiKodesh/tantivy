use std::collections::BTreeSet;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex};

use common::BitSet;
use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::{BitSetDocSet, ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermStreamer, WithoutKeys, ZeroWidthBytes};
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
    // Bytes that do not count towards a term's length when capping.
    zero_width: ZeroWidthBytes,
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
            zero_width: ZeroWidthBytes::NONE,
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
            zero_width: ZeroWidthBytes::NONE,
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
    /// [`set_zero_width_bytes`](Self::set_zero_width_bytes)), then the
    /// highest document count, then byte order.
    ///
    /// Without a rank (and outside JSON paths) the cap is applied INSIDE the
    /// dictionary walk: the dictionary is walked shortest first, level by
    /// level, and the walk stops once the cap is reached, so the long tail of
    /// a broad pattern is never visited. With a rank the whole expansion is
    /// walked and sorted, since the rank is not known before a term is seen.
    ///
    /// The cap is per segment: an index with several segments may use up to
    /// `cap` terms in each. Merging the index into one segment makes the cap
    /// global.
    pub fn set_max_expansions(&mut self, cap: usize) {
        self.max_expansions = Some(cap);
    }

    /// The primary key the cap sorts by (e.g. a fuzzy match's edit distance).
    pub fn set_expansion_rank(&mut self, rank: ExpansionRank) {
        self.expansion_rank = Some(rank);
    }

    /// Bytes that do not count towards a term's length when the cap prefers
    /// shorter terms, e.g. an intra-word apostrophe or quote the tokenizer
    /// keeps. See [`ZeroWidthBytes`].
    pub fn set_zero_width_bytes(&mut self, zero_width: ZeroWidthBytes) {
        self.zero_width = zero_width;
    }

    fn uses_shortest_first_walk(&self) -> bool {
        self.expansion_rank.is_none() && self.json_path_bytes.is_none()
    }

    /// Level-by-level walk of `dict`; stops at `cap`. Ties on the last level
    /// go to the higher document count, then byte order.
    fn cap_shortest_first(&self, dict: &TermDictionary, cap: usize) -> io::Result<Vec<Vec<u8>>> {
        let automaton: &A = &self.automaton;
        let mut walk = dict.search_shortest_first(automaton, self.zero_width);
        let mut kept: Vec<Vec<u8>> = Vec::new();
        while kept.len() < cap {
            let Some((_, level)) = walk.next_level()? else {
                break;
            };
            let room = cap - kept.len();
            if level.len() <= room {
                kept.extend(level.into_iter().map(|(key, _)| key));
            } else {
                let mut level_terms: Vec<(std::cmp::Reverse<u32>, Vec<u8>)> = level
                    .into_iter()
                    .map(|(key, info)| (std::cmp::Reverse(info.doc_freq), key))
                    .collect();
                level_terms.sort_unstable();
                kept.extend(level_terms.into_iter().take(room).map(|(_, key)| key));
            }
        }
        Ok(kept)
    }

    /// Full walk of `dict`, then sort by (rank, shortest, most frequent,
    /// bytes) and keep `cap`.
    fn cap_full_walk(&self, dict: &TermDictionary, cap: usize) -> io::Result<Vec<Vec<u8>>> {
        let mut stream = self.automaton_stream_with_keys(dict)?;
        let mut ranked: Vec<(u32, usize, std::cmp::Reverse<u32>, Vec<u8>)> = Vec::new();
        while stream.advance() {
            let key = stream.key().to_vec();
            let rank = self.expansion_rank.as_ref().map(|r| r(&key)).unwrap_or(0);
            let chars = self.zero_width.term_len(&key);
            ranked.push((rank, chars, std::cmp::Reverse(stream.value().doc_freq), key));
        }
        ranked.sort_unstable();
        ranked.truncate(cap);
        Ok(ranked.into_iter().map(|e| e.3).collect())
    }

    /// Every term the automaton matches in `term_dict`, capped and reported
    /// to the sink as configured.
    fn matched_terms(&self, term_dict: &TermDictionary) -> io::Result<Vec<TermInfo>> {
        if let Some(cap) = self.max_expansions {
            let kept = if self.uses_shortest_first_walk() {
                self.cap_shortest_first(term_dict, cap)?
            } else {
                self.cap_full_walk(term_dict, cap)?
            };
            let mut infos = Vec::with_capacity(kept.len());
            let mut sink = self.matched_terms_sink.as_ref().map(|s| s.lock());
            for key in kept {
                if let Some(info) = term_dict.get(&key)? {
                    infos.push(info);
                    if let Some(sink) = sink.as_mut() {
                        sink.insert(key);
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
}
