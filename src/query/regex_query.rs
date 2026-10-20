use std::clone::Clone;
use std::sync::Arc;

use tantivy_fst::Regex;

use crate::error::TantivyError;
use crate::query::{AutomatonWeight, EnableScoring, MatchedTermsSink, Query, Weight};
use crate::schema::Field;

/// A Regex Query matches all of the documents
/// containing a specific term that matches
/// a regex pattern.
///
/// Wildcard queries (e.g. ho*se) can be achieved
/// by converting them to their regex counterparts.
///
/// ```rust
/// use tantivy::collector::Count;
/// use tantivy::query::RegexQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index, IndexWriter, Term};
///
/// # fn test() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
/// {
///     let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///     index_writer.add_document(doc!(
///         title => "The Name of the Wind",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of Muadib",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "A Dairy Cow",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of a Young Girl",
///     ))?;
///     index_writer.commit()?;
/// }
///
/// let reader = index.reader()?;
/// let searcher = reader.searcher();
///
/// let term = Term::from_field_text(title, "Diary");
/// let query = RegexQuery::from_pattern("d[ai]{2}ry", title)?;
/// let count = searcher.search(&query, &Count)?;
/// assert_eq!(count, 3);
/// Ok(())
/// # }
/// # assert!(test().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct RegexQuery {
    regex: Arc<Regex>,
    field: Field,
    /// When set, every term the regex matches in the term dictionary is
    /// copied into this sink as the search runs.
    matched_terms_sink: Option<MatchedTermsSink>,
    /// When set, at most this many matched terms are used per segment,
    /// shortest first, then most frequent.
    max_expansions: Option<usize>,
}

impl RegexQuery {
    /// Creates a new RegexQuery from a given pattern
    pub fn from_pattern(regex_pattern: &str, field: Field) -> crate::Result<Self> {
        let regex = Regex::new(regex_pattern)
            .map_err(|err| TantivyError::InvalidArgument(format!("RegexQueryError: {err}")))?;
        Ok(RegexQuery::from_regex(regex, field))
    }

    /// Creates a new RegexQuery from a fully built Regex
    pub fn from_regex<T: Into<Arc<Regex>>>(regex: T, field: Field) -> Self {
        RegexQuery {
            regex: regex.into(),
            field,
            matched_terms_sink: None,
            max_expansions: None,
        }
    }

    /// Collects every term this query matches in the term dictionary into
    /// `sink` while the search runs — the concrete words behind the regex
    /// expansion, at the cost of one copy per matched term. Without a sink
    /// the key-less fast path is used, exactly as before.
    pub fn set_matched_terms_sink(&mut self, sink: MatchedTermsSink) {
        self.matched_terms_sink = Some(sink);
    }

    /// Caps the expansion at `cap` terms per segment: the shortest matches
    /// win, then the most frequent. Never an error.
    pub fn set_max_expansions(&mut self, cap: usize) {
        self.max_expansions = Some(cap);
    }

    fn specialized_weight(&self) -> AutomatonWeight<Regex> {
        let mut weight = AutomatonWeight::new(self.field, self.regex.clone());
        if let Some(sink) = &self.matched_terms_sink {
            weight.set_matched_terms_sink(sink.clone());
        }
        if let Some(cap) = self.max_expansions {
            weight.set_max_expansions(cap);
        }
        weight
    }
}

impl Query for RegexQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight()))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use tantivy_fst::Regex;

    use super::RegexQuery;
    use crate::collector::TopDocs;
    use crate::schema::{Field, Schema, TEXT};
    use crate::{assert_nearly_equals, Index, IndexReader, IndexWriter};

    #[test]
    fn test_regex_cap_walks_shortest_first_per_segment() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("t", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        // three segments; in each, the longer term sorts FIRST alphabetically,
        // so an alphabetical walk + cut would keep the wrong one
        for words in [["aaaaa", "ab"], ["aaaa", "abc"], ["aaa", "abcd"]] {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            for w in words {
                writer.add_document(doc!(field => w))?;
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 3);

        let mut query = RegexQuery::from_pattern("a.*", field)?;
        let sink = crate::query::MatchedTermsSink::new();
        query.set_matched_terms_sink(sink.clone());
        query.set_max_expansions(1);
        let count = searcher.search(&query, &crate::collector::Count)?;
        assert_eq!(count, 3);
        // cap 1 per segment: the shortest of each → ab, abc, aaa
        assert_eq!(
            sink.take_strings(),
            vec!["aaa".to_string(), "ab".to_string(), "abc".to_string()]
        );
        Ok(())
    }

    fn build_test_index() -> crate::Result<(IndexReader, Field)> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            index_writer.add_document(doc!(
                country_field => "japan",
            ))?;
            index_writer.add_document(doc!(
                country_field => "korea",
            ))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;

        Ok((reader, country_field))
    }

    fn verify_regex_query(
        query_matching_one: RegexQuery,
        query_matching_zero: RegexQuery,
        reader: IndexReader,
    ) {
        let searcher = reader.searcher();
        {
            let scored_docs = searcher
                .search(
                    &query_matching_one,
                    &TopDocs::with_limit(2).order_by_score(),
                )
                .unwrap();
            assert_eq!(scored_docs.len(), 1, "Expected only 1 document");
            let (score, _) = scored_docs[0];
            assert_nearly_equals!(1.0, score);
        }
        let top_docs = searcher
            .search(
                &query_matching_zero,
                &TopDocs::with_limit(2).order_by_score(),
            )
            .unwrap();
        assert!(top_docs.is_empty(), "Expected ZERO document");
    }

    #[test]
    pub fn test_regex_query() -> crate::Result<()> {
        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_pattern("jap[ao]n", field)?;
        let matching_zero = RegexQuery::from_pattern("jap[A-Z]n", field)?;
        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_construct_from_regex() -> crate::Result<()> {
        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_regex(Regex::new("jap[ao]n").unwrap(), field);
        let matching_zero = RegexQuery::from_regex(Regex::new("jap[A-Z]n").unwrap(), field);

        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_construct_from_reused_regex() -> crate::Result<()> {
        let r1 = Arc::new(Regex::new("jap[ao]n").unwrap());
        let r2 = Arc::new(Regex::new("jap[A-Z]n").unwrap());

        let (reader, field) = build_test_index()?;

        let matching_one = RegexQuery::from_regex(r1.clone(), field);
        let matching_zero = RegexQuery::from_regex(r2.clone(), field);

        verify_regex_query(matching_one, matching_zero, reader.clone());

        let matching_one = RegexQuery::from_regex(r1, field);
        let matching_zero = RegexQuery::from_regex(r2, field);

        verify_regex_query(matching_one, matching_zero, reader);
        Ok(())
    }

    #[test]
    pub fn test_pattern_error() {
        let (_reader, field) = build_test_index().unwrap();

        match RegexQuery::from_pattern(r"(foo", field) {
            Err(crate::TantivyError::InvalidArgument(msg)) => {
                assert!(msg.contains("error: unclosed group"))
            }
            res => panic!("unexpected result: {res:?}"),
        }
    }
}
