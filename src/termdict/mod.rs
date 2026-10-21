//! The term dictionary main role is to associate the sorted [`Term`s](crate::Term) to
//! a [`TermInfo`] struct that contains some meta-information
//! about the term.
//!
//! Internally, the term dictionary relies on the `fst` crate to store
//! a sorted mapping that associate each term to its rank in the lexicographical order.
//! For instance, in a dictionary containing the sorted terms "abba", "bjork", "blur" and "donovan",
//! the [`TermOrdinal`] are respectively `0`, `1`, `2`, and `3`.
//!
//! For `u64`-terms, tantivy explicitly uses a `BigEndian` representation to ensure that the
//! lexicographical order matches the natural order of integers.
//!
//! `i64`-terms are transformed to `u64` using a continuous mapping `val ⟶ val - i64::MIN`
//! and then treated as a `u64`.
//!
//! `f64`-terms are transformed to `u64` using a mapping that preserve order, and are then treated
//! as `u64`.
//!
//! A second datastructure makes it possible to access a [`TermInfo`].

#[cfg(not(feature = "quickwit"))]
mod fst_termdict;
#[cfg(not(feature = "quickwit"))]
use fst_termdict as termdict;

#[cfg(feature = "quickwit")]
mod sstable_termdict;
#[cfg(feature = "quickwit")]
use sstable_termdict as termdict;

#[cfg(test)]
mod tests;

/// Position of the term in the sorted list of terms.
pub type TermOrdinal = u64;

use std::io;

use common::file_slice::FileSlice;
use common::BinarySerializable;
pub use common::{KeyTracking, WithoutKeys};
use tantivy_fst::Automaton;

use self::termdict::{
    TermDictionary as InnerTermDict, TermDictionaryBuilder as InnerTermDictBuilder,
    TermStreamerBuilder,
};
pub use self::termdict::{TermMerger, TermStreamer};
use crate::postings::TermInfo;

#[derive(Debug, Eq, PartialEq)]
#[repr(u32)]
enum DictionaryType {
    Fst = 1,
    SSTable = 2,
}

impl TryFrom<u32> for DictionaryType {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(DictionaryType::Fst),
            2 => Ok(DictionaryType::SSTable),
            _ => Err("Invalid value for DictionaryType"),
        }
    }
}

#[cfg(not(feature = "quickwit"))]
const CURRENT_TYPE: DictionaryType = DictionaryType::Fst;

#[cfg(feature = "quickwit")]
const CURRENT_TYPE: DictionaryType = DictionaryType::SSTable;

// TODO in the future this should become an enum of supported dictionaries
/// A TermDictionary wrapping either an FST based dictionary or a SSTable based one.
#[derive(Clone)]
pub struct TermDictionary(InnerTermDict);

impl TermDictionary {
    /// Opens a `TermDictionary`.
    pub fn open(file: FileSlice) -> io::Result<Self> {
        let (main_slice, dict_type) = file.split_from_end(4);
        let mut dict_type = dict_type.read_bytes()?;
        let dict_type = u32::deserialize(&mut dict_type)?;
        let dict_type = DictionaryType::try_from(dict_type).map_err(|_| {
            io::Error::other(format!("Unsupported dictionary type, found {dict_type}"))
        })?;

        if dict_type != CURRENT_TYPE {
            return Err(io::Error::other(format!(
                "Unsupported dictionary type, compiled tantivy with {CURRENT_TYPE:?}, but got \
                 {dict_type:?}",
            )));
        }

        InnerTermDict::open(main_slice).map(TermDictionary)
    }

    /// Creates an empty term dictionary which contains no terms.
    pub fn empty() -> Self {
        TermDictionary(InnerTermDict::empty())
    }

    /// Returns the number of terms in the dictionary.
    /// Term ordinals range from 0 to `num_terms() - 1`.
    pub fn num_terms(&self) -> usize {
        self.0.num_terms()
    }

    /// Returns the ordinal associated with a given term.
    pub fn term_ord<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermOrdinal>> {
        self.0.term_ord(key)
    }

    /// Stores the term associated with a given term ordinal in
    /// a `bytes` buffer.
    ///
    /// Term ordinals are defined as the position of the term in
    /// the sorted list of terms.
    ///
    /// Returns true if and only if the term has been found.
    ///
    /// Regardless of whether the term is found or not,
    /// the buffer may be modified.
    pub fn ord_to_term(&self, ord: TermOrdinal, bytes: &mut Vec<u8>) -> io::Result<bool> {
        self.0.ord_to_term(ord, bytes)
    }

    // this isn't used, and has different prototype in Fst and SSTable
    // Returns the number of terms in the dictionary.
    // pub fn term_info_from_ord(&self, term_ord: TermOrdinal) -> TermInfo {
    // self.0.term_info_from_ord(term_ord)
    // }

    /// Lookups the value corresponding to the key.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermInfo>> {
        self.0.get(key)
    }

    /// Walks the automaton's matches shortest first, one length level (in
    /// characters, see [`ZeroWidthChars`]) at a time. Stop pulling levels
    /// when enough terms are collected.
    ///
    /// With the FST dictionary this is a breadth-first walk of the FST and
    /// the levels that are never pulled are never visited. The SSTable
    /// dictionary has no such structure: it walks the whole expansion once
    /// and groups it by length, so the result is the same but the cost is
    /// that of a full walk.
    pub fn search_shortest_first<'a, A>(
        &'a self,
        automaton: &'a A,
        zero_width: ZeroWidthChars,
    ) -> ShortestFirstTerms<'a, A>
    where
        A: Automaton,
        A::State: Clone,
    {
        ShortestFirstTerms::new(&self.0, automaton, zero_width)
    }

    /// Returns a range builder, to stream all of the terms
    /// within an interval.
    pub fn range(&self) -> TermStreamerBuilder<'_> {
        self.0.range()
    }

    /// A stream of all the sorted terms.
    pub fn stream(&self) -> io::Result<TermStreamer<'_>> {
        self.0.stream()
    }

    /// Returns a search builder, to stream all of the terms
    /// within the Automaton
    pub fn search<'a, A: Automaton + 'a>(&'a self, automaton: A) -> TermStreamerBuilder<'a, A>
    where
        A::State: Clone,
    {
        self.0.search(automaton)
    }

    #[cfg(feature = "quickwit")]
    /// Lookups the value corresponding to the key.
    pub async fn get_async<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermInfo>> {
        self.0.get_async(key).await
    }

    #[cfg(feature = "quickwit")]
    #[doc(hidden)]
    pub async fn warm_up_dictionary(&self) -> io::Result<()> {
        self.0.warm_up_dictionary().await
    }

    #[cfg(feature = "quickwit")]
    /// Returns a file slice covering a set of sstable blocks
    /// that includes the key range passed in arguments.
    pub fn file_slice_for_range(
        &self,
        key_range: impl std::ops::RangeBounds<[u8]>,
        limit: Option<u64>,
    ) -> FileSlice {
        self.0.file_slice_for_range(key_range, limit)
    }
}

/// A TermDictionaryBuilder wrapping either an FST or a SSTable dictionary builder.
pub struct TermDictionaryBuilder<W: io::Write>(InnerTermDictBuilder<W>);

impl<W: io::Write> TermDictionaryBuilder<W> {
    /// Creates a new `TermDictionaryBuilder`
    pub fn create(w: W) -> io::Result<Self> {
        InnerTermDictBuilder::create(w).map(TermDictionaryBuilder)
    }

    /// Inserts a `(key, value)` pair in the term dictionary.
    ///
    /// *Keys have to be inserted in order.*
    pub fn insert<K: AsRef<[u8]>>(&mut self, key_ref: K, value: &TermInfo) -> io::Result<()> {
        self.0.insert(key_ref, value)
    }

    /// # Warning
    /// Horribly dangerous internal API
    ///
    /// If used, it must be used by systematically alternating calls
    /// to insert_key and insert_value.
    ///
    /// Prefer using `.insert(key, value)`
    pub fn insert_key(&mut self, key: &[u8]) -> io::Result<()> {
        self.0.insert_key(key)
    }

    /// # Warning
    ///
    /// Horribly dangerous internal API. See `.insert_key(...)`.
    pub fn insert_value(&mut self, term_info: &TermInfo) -> io::Result<()> {
        self.0.insert_value(term_info)
    }

    /// Finalize writing the builder, and returns the underlying
    /// `Write` object.
    pub fn finish(self) -> io::Result<W> {
        let mut writer = self.0.finish()?;
        (CURRENT_TYPE as u32).serialize(&mut writer)?;
        Ok(writer)
    }
}

/// Characters that do not count towards a term's length in
/// [`TermDictionary::search_shortest_first`] and the expansion cap of
/// [`AutomatonWeight`](crate::query::AutomatonWeight): an intra-word mark a
/// tokenizer keeps inside its terms (an apostrophe, a quote, a combining
/// accent or vowel point) that should not make the term "longer" than the
/// same word without it.
///
/// ASCII bytes are given as a set ([`ZeroWidthChars::new`]); other
/// characters as inclusive code point ranges
/// ([`ZeroWidthChars::with_ranges`]). Terms are read as UTF-8: a byte that
/// does not start a valid character is never zero-width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZeroWidthChars {
    ascii: [u64; 2],
    ranges: &'static [(char, char)],
}

impl ZeroWidthChars {
    /// Every character counts.
    pub const NONE: ZeroWidthChars = ZeroWidthChars {
        ascii: [0; 2],
        ranges: &[],
    };

    /// The given ASCII bytes are zero-width.
    ///
    /// # Panics
    /// If a byte is not ASCII.
    pub const fn new(bytes: &[u8]) -> ZeroWidthChars {
        let mut bits = [0u64; 2];
        let mut i = 0;
        while i < bytes.len() {
            let byte = bytes[i];
            assert!(byte.is_ascii(), "zero-width bytes must be ASCII");
            bits[(byte >> 6) as usize] |= 1u64 << (byte & 63);
            i += 1;
        }
        ZeroWidthChars {
            ascii: bits,
            ranges: &[],
        }
    }

    /// These inclusive ranges of characters are zero-width too (replacing
    /// any ranges set before). ASCII characters in them are ignored: use
    /// [`ZeroWidthChars::new`] for those.
    ///
    /// The ranges are `'static` so the set stays `Copy` and can be built in
    /// a `const`; they are scanned linearly, so keep the list short (a few
    /// ranges of marks).
    pub const fn with_ranges(self, ranges: &'static [(char, char)]) -> ZeroWidthChars {
        ZeroWidthChars {
            ascii: self.ascii,
            ranges,
        }
    }

    /// Whether the ASCII `byte` is zero-width (always false for a byte of a
    /// multi-byte character: see [`ZeroWidthChars::contains`]).
    #[inline]
    pub fn contains_byte(&self, byte: u8) -> bool {
        byte.is_ascii() && (self.ascii[(byte >> 6) as usize] >> (byte & 63)) & 1 == 1
    }

    /// Whether the character `c` is zero-width.
    #[inline]
    pub fn contains(&self, c: char) -> bool {
        if c.is_ascii() {
            return self.contains_byte(c as u8);
        }
        self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi)
    }

    /// Whether some multi-byte characters are zero-width.
    #[inline]
    pub fn has_multibyte(&self) -> bool {
        !self.ranges.is_empty()
    }

    /// Whether no character is zero-width.
    pub fn is_empty(&self) -> bool {
        self.ascii == [0; 2] && self.ranges.is_empty()
    }

    /// For the UTF-8 character starting at `key[at]` (not ASCII, not a
    /// continuation byte): whether it is zero-width, and its length in bytes
    /// (1 when it is not a valid character).
    #[inline]
    fn multibyte_at(&self, key: &[u8], at: usize) -> (bool, usize) {
        let len = match key[at] {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return (false, 1),
        };
        let Some(bytes) = key.get(at..at + len) else {
            return (false, 1);
        };
        match std::str::from_utf8(bytes) {
            Ok(s) => (s.chars().next().is_some_and(|c| self.contains(c)), len),
            Err(_) => (false, 1),
        }
    }

    /// Whether the last `len` bytes of `key`, one complete character, are a
    /// zero-width character.
    #[inline]
    pub(crate) fn ends_with_zero_width(&self, key: &[u8], len: usize) -> bool {
        if len == 1 {
            return key.last().is_some_and(|&b| self.contains_byte(b));
        }
        self.has_multibyte()
            && key.len() >= len
            && self.multibyte_at(key, key.len() - len) == (true, len)
    }

    /// The length of a term in characters: UTF-8 characters, not counting
    /// the zero-width ones.
    pub fn term_len(&self, key: &[u8]) -> usize {
        if !self.has_multibyte() {
            return key
                .iter()
                .filter(|&&b| b & 0xC0 != 0x80 && !self.contains_byte(b))
                .count();
        }
        let mut count = 0;
        let mut at = 0;
        while at < key.len() {
            let b = key[at];
            if b.is_ascii() {
                count += usize::from(!self.contains_byte(b));
                at += 1;
            } else if b & 0xC0 == 0x80 {
                at += 1; // a stray continuation byte is not a character
            } else {
                let (zero, len) = self.multibyte_at(key, at);
                count += usize::from(!zero);
                at += len;
            }
        }
        count
    }

    /// The term without its zero-width characters: terms that differ only
    /// in them share this form (and therefore one expansion slot under a
    /// cap, see `AutomatonWeight::set_max_expansions`).
    pub fn strip<'a>(&self, key: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        if self.is_empty() {
            return std::borrow::Cow::Borrowed(key);
        }
        if !self.has_multibyte() {
            if !key.iter().any(|&b| self.contains_byte(b)) {
                return std::borrow::Cow::Borrowed(key);
            }
            return std::borrow::Cow::Owned(
                key.iter()
                    .copied()
                    .filter(|&b| !self.contains_byte(b))
                    .collect(),
            );
        }
        let mut out: Option<Vec<u8>> = None;
        let mut at = 0;
        while at < key.len() {
            let b = key[at];
            let (zero, len) = if b.is_ascii() {
                (self.contains_byte(b), 1)
            } else if b & 0xC0 == 0x80 {
                (false, 1)
            } else {
                self.multibyte_at(key, at)
            };
            match (&mut out, zero) {
                (None, true) => out = Some(key[..at].to_vec()),
                (Some(out), false) => out.extend_from_slice(&key[at..at + len]),
                _ => {}
            }
            at += len;
        }
        match out {
            Some(out) => std::borrow::Cow::Owned(out),
            None => std::borrow::Cow::Borrowed(key),
        }
    }
}

/// One level of a shortest-first walk: every matching term of one length,
/// with its [`TermInfo`], in byte order.
pub type ShortestFirstLevel = Vec<(Vec<u8>, TermInfo)>;

/// Shortest-first walk over a [`TermDictionary`]: see
/// [`TermDictionary::search_shortest_first`].
#[cfg(not(feature = "quickwit"))]
pub struct ShortestFirstTerms<'a, A: Automaton> {
    dict: &'a fst_termdict::TermDictionary,
    walk: fst_termdict::shortest_first::ShortestFirstWalk<'a, A>,
}

#[cfg(not(feature = "quickwit"))]
impl<'a, A> ShortestFirstTerms<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    fn new(
        dict: &'a fst_termdict::TermDictionary,
        automaton: &'a A,
        zero_width: ZeroWidthChars,
    ) -> Self {
        ShortestFirstTerms {
            dict,
            walk: dict.search_shortest_first(automaton, zero_width),
        }
    }

    /// The next non-empty level: (length in characters, its terms). `None`
    /// when the walk is exhausted.
    pub fn next_level(&mut self) -> io::Result<Option<(usize, ShortestFirstLevel)>> {
        let Some(level) = self.walk.next() else {
            return Ok(None);
        };
        let length = self.walk.level() - 1;
        let dict = self.dict;
        let terms = level
            .into_iter()
            .map(|(key, ord)| (key, dict.term_info_from_ord(ord)))
            .collect();
        Ok(Some((length, terms)))
    }

    /// True when a later [`next_level`](Self::next_level) call would return
    /// a level: searches depth first for one more match (stops at the first).
    pub fn has_more(&mut self) -> io::Result<bool> {
        Ok(self.walk.has_more())
    }
}

/// Shortest-first walk over a [`TermDictionary`]: see
/// [`TermDictionary::search_shortest_first`].
///
/// The SSTable dictionary cannot be walked breadth-first; the first call
/// streams the whole expansion and groups it by length.
#[cfg(feature = "quickwit")]
pub struct ShortestFirstTerms<'a, A: Automaton> {
    dict: &'a sstable_termdict::TermDictionary,
    automaton: &'a A,
    zero_width: ZeroWidthChars,
    levels: Option<std::vec::IntoIter<(usize, ShortestFirstLevel)>>,
}

#[cfg(feature = "quickwit")]
impl<'a, A> ShortestFirstTerms<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    fn new(
        dict: &'a sstable_termdict::TermDictionary,
        automaton: &'a A,
        zero_width: ZeroWidthChars,
    ) -> Self {
        ShortestFirstTerms {
            dict,
            automaton,
            zero_width,
            levels: None,
        }
    }

    /// The next non-empty level: (length in characters, its terms). `None`
    /// when the walk is exhausted.
    pub fn next_level(&mut self) -> io::Result<Option<(usize, ShortestFirstLevel)>> {
        if self.levels.is_none() {
            let mut by_len: std::collections::BTreeMap<usize, ShortestFirstLevel> =
                std::collections::BTreeMap::new();
            let mut stream = self.dict.search(self.automaton).into_stream()?;
            while stream.advance() {
                let key = stream.key().to_vec();
                let len = self.zero_width.term_len(&key);
                by_len
                    .entry(len)
                    .or_default()
                    .push((key, stream.value().clone()));
            }
            self.levels = Some(by_len.into_iter().collect::<Vec<_>>().into_iter());
        }
        Ok(self.levels.as_mut().unwrap().next())
    }

    /// True when a later [`next_level`](Self::next_level) call would return
    /// a level.
    pub fn has_more(&mut self) -> io::Result<bool> {
        if self.levels.is_none() {
            let Some(first) = self.next_level()? else {
                return Ok(false);
            };
            let rest: Vec<_> = self.levels.take().unwrap().collect();
            self.levels = Some(
                std::iter::once(first)
                    .chain(rest)
                    .collect::<Vec<_>>()
                    .into_iter(),
            );
            return Ok(true);
        }
        Ok(self.levels.as_ref().unwrap().len() > 0)
    }
}
