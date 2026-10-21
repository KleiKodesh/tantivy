use std::path::PathBuf;
use std::{io, str};

use super::{TermDictionary, TermDictionaryBuilder, TermStreamer, ZeroWidthChars};
use crate::directory::{Directory, FileSlice, RamDirectory, TerminatingWrite};
use crate::postings::TermInfo;

const BLOCK_SIZE: usize = 1_500;

fn make_term_info(term_ord: u64) -> TermInfo {
    let offset = |term_ord: u64| (term_ord * 100 + term_ord * term_ord) as usize;
    TermInfo {
        doc_freq: term_ord as u32,
        postings_range: offset(term_ord)..offset(term_ord + 1),
        positions_range: offset(term_ord) * 2..offset(term_ord + 1) * 2,
    }
}

#[test]
fn test_empty_term_dictionary() {
    let empty = TermDictionary::empty();
    assert!(empty.stream().unwrap().next().is_none());
}

#[test]
fn test_term_ordinals() -> crate::Result<()> {
    const COUNTRIES: [&str; 7] = [
        "San Marino",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
    ];
    let directory = RamDirectory::create();
    let path = PathBuf::from("TermDictionary");
    {
        let write = directory.open_write(&path)?;
        let mut term_dictionary_builder = TermDictionaryBuilder::create(write)?;
        for term in COUNTRIES.iter() {
            term_dictionary_builder.insert(term.as_bytes(), &make_term_info(0u64))?;
        }
        term_dictionary_builder.finish()?.terminate()?;
    }
    let term_file = directory.open_read(&path)?;
    let term_dict: TermDictionary = TermDictionary::open(term_file)?;
    for (term_ord, term) in COUNTRIES.iter().enumerate() {
        assert_eq!(term_dict.term_ord(term)?, Some(term_ord as u64));
        let mut bytes = vec![];
        assert!(term_dict.ord_to_term(term_ord as u64, &mut bytes)?);
        assert_eq!(bytes, term.as_bytes());
    }
    Ok(())
}

#[test]
fn test_term_dictionary_simple() -> crate::Result<()> {
    let directory = RamDirectory::create();
    let path = PathBuf::from("TermDictionary");
    {
        let write = directory.open_write(&path)?;
        let mut term_dictionary_builder = TermDictionaryBuilder::create(write)?;
        term_dictionary_builder.insert("abc".as_bytes(), &make_term_info(34u64))?;
        term_dictionary_builder.insert("abcd".as_bytes(), &make_term_info(346u64))?;
        term_dictionary_builder.finish()?.terminate()?;
    }
    let file = directory.open_read(&path)?;
    let term_dict: TermDictionary = TermDictionary::open(file)?;
    assert_eq!(term_dict.get("abc")?.unwrap().doc_freq, 34u32);
    assert_eq!(term_dict.get("abcd")?.unwrap().doc_freq, 346u32);
    let mut stream = term_dict.stream()?;
    {
        {
            let (k, v) = stream.next().unwrap();
            assert_eq!(k, "abc".as_bytes());
            assert_eq!(v.doc_freq, 34u32);
        }
        assert_eq!(stream.key(), "abc".as_bytes());
        assert_eq!(stream.value().doc_freq, 34u32);
    }
    {
        {
            let (k, v) = stream.next().unwrap();
            assert_eq!(k, "abcd".as_bytes());
            assert_eq!(v.doc_freq, 346u32);
        }
        assert_eq!(stream.key(), "abcd".as_bytes());
        assert_eq!(stream.value().doc_freq, 346u32);
    }
    assert!(!stream.advance());
    Ok(())
}

#[test]
fn test_term_dictionary_stream() -> crate::Result<()> {
    let ids: Vec<_> = (0u32..10_000u32)
        .map(|i| (format!("doc{i:0>6}"), i))
        .collect();
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(vec![]).unwrap();
        for (id, i) in &ids {
            term_dictionary_builder
                .insert(id.as_bytes(), &make_term_info(*i as u64))
                .unwrap();
        }
        term_dictionary_builder.finish()?
    };
    let term_file = FileSlice::from(buffer);
    let term_dictionary: TermDictionary = TermDictionary::open(term_file)?;
    {
        let mut streamer = term_dictionary.stream()?;
        let mut i = 0;
        while let Some((streamer_k, streamer_v)) = streamer.next() {
            let (key, v) = &ids[i];
            assert_eq!(streamer_k, key.as_bytes());
            assert_eq!(streamer_v, &make_term_info(*v as u64));
            i += 1;
        }
    }

    let (key, val) = &ids[2047];
    assert_eq!(
        term_dictionary.get(key.as_bytes())?,
        Some(make_term_info(*val as u64))
    );
    Ok(())
}

#[test]
fn test_stream_high_range_prefix_suffix() -> crate::Result<()> {
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(vec![]).unwrap();
        // term requires more than 16bits
        term_dictionary_builder.insert("abcdefghijklmnopqrstuvwxy", &make_term_info(1))?;
        term_dictionary_builder.insert("abcdefghijklmnopqrstuvwxyz", &make_term_info(2))?;
        term_dictionary_builder.insert("abr", &make_term_info(3))?;
        term_dictionary_builder.finish()?
    };
    let term_dict_file = FileSlice::from(buffer);
    let term_dictionary: TermDictionary = TermDictionary::open(term_dict_file)?;
    let mut kv_stream = term_dictionary.stream()?;
    assert!(kv_stream.advance());
    assert_eq!(kv_stream.key(), "abcdefghijklmnopqrstuvwxy".as_bytes());
    assert_eq!(kv_stream.value(), &make_term_info(1));
    assert!(kv_stream.advance());
    assert_eq!(kv_stream.key(), "abcdefghijklmnopqrstuvwxyz".as_bytes());
    assert_eq!(kv_stream.value(), &make_term_info(2));
    assert!(kv_stream.advance());
    assert_eq!(kv_stream.key(), "abr".as_bytes());
    assert_eq!(kv_stream.value(), &make_term_info(3));
    assert!(!kv_stream.advance());
    Ok(())
}

#[test]
fn test_stream_range() -> crate::Result<()> {
    let ids: Vec<_> = (0u32..10_000u32)
        .map(|i| (format!("doc{i:0>6}"), i))
        .collect();
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(vec![]).unwrap();
        for (id, i) in &ids {
            term_dictionary_builder
                .insert(id.as_bytes(), &make_term_info(*i as u64))
                .unwrap();
        }
        term_dictionary_builder.finish()?
    };

    let file = FileSlice::from(buffer);

    let term_dictionary: TermDictionary = TermDictionary::open(file)?;
    {
        for i in (0..20).chain(6000..8_000) {
            let (target_key, _) = &ids[i];
            let mut streamer = term_dictionary
                .range()
                .ge(target_key.as_bytes())
                .into_stream()?;
            for j in 0..3 {
                let (streamer_k, streamer_v) = streamer.next().unwrap();
                let (key, v) = &ids[i + j];
                assert_eq!(str::from_utf8(streamer_k).unwrap(), key);
                assert_eq!(streamer_v.doc_freq, *v);
                assert_eq!(streamer_v, &make_term_info(*v as u64));
            }
        }
    }

    {
        for i in (0..20).chain(BLOCK_SIZE - 10..BLOCK_SIZE + 10) {
            let (target_key, _) = &ids[i];
            let mut streamer = term_dictionary
                .range()
                .gt(target_key.as_bytes())
                .into_stream()?;
            for j in 0..3 {
                let (streamer_k, streamer_v) = streamer.next().unwrap();
                let (key, v) = &ids[i + j + 1];
                assert_eq!(streamer_k, key.as_bytes());
                assert_eq!(streamer_v.doc_freq, *v);
            }
        }
    }

    {
        for i in (0..20).chain(BLOCK_SIZE - 10..BLOCK_SIZE + 10) {
            for j in 0..3 {
                let (fst_key, _) = &ids[i];
                let (last_key, _) = &ids[i + j];
                let mut streamer = term_dictionary
                    .range()
                    .ge(fst_key.as_bytes())
                    .lt(last_key.as_bytes())
                    .into_stream()?;
                for _ in 0..j {
                    assert!(streamer.next().is_some());
                }
                assert!(streamer.next().is_none());
            }
        }
    }
    Ok(())
}

#[test]
fn test_empty_string() -> crate::Result<()> {
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(vec![]).unwrap();
        term_dictionary_builder
            .insert([], &make_term_info(1_u64))
            .unwrap();
        term_dictionary_builder
            .insert([1u8], &make_term_info(2_u64))
            .unwrap();
        term_dictionary_builder.finish()?
    };
    let file = FileSlice::from(buffer);
    let term_dictionary: TermDictionary = TermDictionary::open(file)?;
    let mut stream = term_dictionary.stream()?;
    assert!(stream.advance());
    assert!(stream.key().is_empty());
    assert!(stream.advance());
    assert_eq!(stream.key(), &[1u8]);
    assert!(!stream.advance());
    Ok(())
}

fn stream_range_test_dict() -> io::Result<TermDictionary> {
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(Vec::new())?;
        for i in 0u8..10u8 {
            let number_arr = [i; 1];
            term_dictionary_builder.insert(number_arr, &make_term_info(i as u64))?;
        }
        term_dictionary_builder.finish()?
    };
    let file = FileSlice::from(buffer);
    TermDictionary::open(file)
}

#[test]
fn test_stream_range_boundaries_forward() -> crate::Result<()> {
    let term_dictionary = stream_range_test_dict()?;
    let value_list = |mut streamer: TermStreamer<'_>| {
        let mut res: Vec<u32> = vec![];
        while let Some((_, v)) = streamer.next() {
            res.push(v.doc_freq);
        }
        res
    };
    {
        let range = term_dictionary.range().ge([2u8]).into_stream()?;
        assert_eq!(
            value_list(range),
            vec![2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32]
        );
    }
    {
        let range = term_dictionary.range().gt([2u8]).into_stream()?;
        assert_eq!(
            value_list(range),
            vec![3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32]
        );
    }
    {
        let range = term_dictionary.range().lt([6u8]).into_stream()?;
        assert_eq!(value_list(range), vec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32]);
    }
    {
        let range = term_dictionary.range().le([6u8]).into_stream()?;
        assert_eq!(
            value_list(range),
            vec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32]
        );
    }
    {
        let range = term_dictionary.range().ge([0u8]).lt([5u8]).into_stream()?;
        assert_eq!(value_list(range), vec![0u32, 1u32, 2u32, 3u32, 4u32]);
    }
    Ok(())
}

#[cfg(not(feature = "quickwit"))]
#[test]
fn test_stream_range_boundaries_backward() -> crate::Result<()> {
    let term_dictionary = stream_range_test_dict()?;
    let value_list_backward = |mut streamer: TermStreamer<'_>| {
        let mut res: Vec<u32> = vec![];
        while let Some((_, v)) = streamer.next() {
            res.push(v.doc_freq);
        }
        res.reverse();
        res
    };
    {
        let range = term_dictionary.range().backward().into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32]
        );
    }
    {
        let range = term_dictionary.range().ge([2u8]).backward().into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32]
        );
    }
    {
        let range = term_dictionary.range().gt([2u8]).backward().into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32]
        );
    }
    {
        let range = term_dictionary.range().lt([6u8]).backward().into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32]
        );
    }
    {
        let range = term_dictionary.range().le([6u8]).backward().into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32]
        );
    }
    {
        let range = term_dictionary
            .range()
            .ge([0u8])
            .lt([5u8])
            .backward()
            .into_stream()?;
        assert_eq!(
            value_list_backward(range),
            vec![0u32, 1u32, 2u32, 3u32, 4u32]
        );
    }
    Ok(())
}

#[test]
fn test_ord_to_term() -> crate::Result<()> {
    let termdict = stream_range_test_dict()?;
    let mut bytes = vec![];
    for b in 0u8..10u8 {
        termdict.ord_to_term(b as u64, &mut bytes)?;
        assert_eq!(&bytes, &[b]);
    }
    Ok(())
}

#[test]
fn test_stream_term_ord() -> crate::Result<()> {
    let termdict = stream_range_test_dict()?;
    let mut stream = termdict.stream()?;
    for b in 0u8..10u8 {
        assert!(stream.advance());
        assert_eq!(stream.term_ord(), b as u64);
        assert_eq!(stream.key(), &[b]);
    }
    assert!(!stream.advance());
    Ok(())
}

#[test]
fn test_automaton_search() -> crate::Result<()> {
    use levenshtein_automata::LevenshteinAutomatonBuilder;

    use crate::query::DfaWrapper;

    const COUNTRIES: [&str; 7] = [
        "San Marino",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
    ];

    let directory = RamDirectory::create();
    let path = PathBuf::from("TermDictionary");
    {
        let write = directory.open_write(&path)?;
        let mut term_dictionary_builder = TermDictionaryBuilder::create(write)?;
        for term in COUNTRIES.iter() {
            term_dictionary_builder.insert(term.as_bytes(), &make_term_info(0u64))?;
        }
        term_dictionary_builder.finish()?.terminate()?;
    }
    let file = directory.open_read(&path)?;
    let term_dict: TermDictionary = TermDictionary::open(file)?;

    // We can now build an entire dfa.
    let lev_automaton_builder = LevenshteinAutomatonBuilder::new(2, true);
    let automaton = DfaWrapper(lev_automaton_builder.build_dfa("Spaen"));

    let mut range = term_dict.search(automaton).into_stream()?;

    // get the first finding
    assert!(range.advance());
    assert_eq!("Spain".as_bytes(), range.key());
    assert!(!range.advance());
    Ok(())
}

#[test]
fn test_range_without_keys() {
    let term_dict = stream_range_test_dict().unwrap();
    let mut stream = term_dict
        .range()
        .ge([2u8])
        .lt([5u8])
        .without_keys()
        .into_stream()
        .unwrap();
    for term_ord in 2u64..5u64 {
        assert!(stream.advance());
        assert_eq!(stream.term_ord(), term_ord);
        assert_eq!(stream.value(), &make_term_info(term_ord));
    }
    assert!(!stream.advance());
}

#[test]
fn test_search_without_keys() {
    const COUNTRIES: [&str; 7] = [
        "San Marino",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
    ];
    let buffer: Vec<u8> = {
        let mut term_dictionary_builder = TermDictionaryBuilder::create(Vec::new()).unwrap();
        for (term_ord, term) in COUNTRIES.iter().enumerate() {
            term_dictionary_builder
                .insert(term.as_bytes(), &make_term_info(term_ord as u64))
                .unwrap();
        }
        term_dictionary_builder.finish().unwrap()
    };
    let term_dict = TermDictionary::open(FileSlice::from(buffer)).unwrap();
    let regex = tantivy_fst::Regex::new("S[lw].*").unwrap();
    let mut stream = term_dict
        .search(regex)
        .without_keys()
        .into_stream()
        .unwrap();
    for term_ord in [2u64, 3, 5, 6] {
        assert!(stream.advance());
        assert_eq!(stream.term_ord(), term_ord);
        assert_eq!(stream.value(), &make_term_info(term_ord));
    }
    assert!(!stream.advance());
}

#[test]
fn test_zero_width_chars() {
    let none = ZeroWidthChars::NONE;
    assert!(none.is_empty());
    assert!(!none.contains_byte(b'"'));
    assert_eq!(none.term_len("a\"b\u{e9}".as_bytes()), 4);

    let quote = ZeroWidthChars::new(b"\"'");
    assert!(!quote.is_empty());
    assert!(quote.contains_byte(b'"'));
    assert!(quote.contains('\''));
    assert!(!quote.contains_byte(b'a'));
    assert!(!quote.contains_byte(0xC3));
    assert!(!quote.has_multibyte());
    assert_eq!(quote.term_len("a\"b\u{e9}".as_bytes()), 3);
    assert_eq!(quote.term_len(b""), 0);
    assert_eq!(quote.strip(b"a\"b").as_ref(), b"ab");
    assert_eq!(ZeroWidthChars::default(), none);

    // multi-byte characters by range: combining marks (2 bytes) and a
    // 3-byte range
    static RANGES: [(char, char); 2] = [('\u{300}', '\u{36F}'), ('\u{200B}', '\u{200D}')];
    let marks = ZeroWidthChars::new(b"\"").with_ranges(&RANGES);
    assert!(marks.has_multibyte() && !marks.is_empty());
    assert!(marks.contains('\u{301}') && marks.contains('\u{200C}') && marks.contains('"'));
    assert!(!marks.contains('\u{e9}') && !marks.contains('a'));
    let term = "e\u{301}t\u{200C}e\"\u{e9}";
    assert_eq!(marks.term_len(term.as_bytes()), 4);
    assert_eq!(
        marks.strip(term.as_bytes()).as_ref(),
        "ete\u{e9}".as_bytes()
    );
    assert!(matches!(marks.strip(b"abc"), std::borrow::Cow::Borrowed(_)));
    // bytes that are not valid UTF-8 are never zero-width
    assert_eq!(marks.term_len(&[b'a', 0xCC]), 2);
    assert_eq!(marks.strip(&[0xCC, 0x81, 0xCC]).as_ref(), &[0xCC]);
    assert_eq!(
        ZeroWidthChars::NONE
            .with_ranges(&RANGES)
            .term_len("\u{301}".as_bytes()),
        0
    );
}

#[test]
fn test_search_shortest_first_multibyte_zero_width() -> crate::Result<()> {
    // a 3-byte zero-width char at the start of a term and two zero-width
    // chars in a row; a 4-byte char that counts
    static RANGES: [(char, char); 2] = [('\u{300}', '\u{36F}'), ('\u{200B}', '\u{200D}')];
    let terms: &[&str] = &["ab", "a\u{200C}\u{301}b", "\u{200C}a", "\u{1F600}"];
    let buffer: Vec<u8> = {
        let mut builder = TermDictionaryBuilder::create(Vec::new())?;
        for (ord, term) in terms.iter().enumerate() {
            builder.insert(term.as_bytes(), &make_term_info(ord as u64))?;
        }
        builder.finish()?
    };
    let dict = TermDictionary::open(FileSlice::from(buffer))?;
    let zero_width = ZeroWidthChars::NONE.with_ranges(&RANGES);
    let mut walk = dict.search_shortest_first(&tantivy_fst::automaton::AlwaysMatch, zero_width);
    let mut levels = Vec::new();
    while let Some((len, level)) = walk.next_level()? {
        let keys: Vec<String> = level
            .into_iter()
            .map(|(key, _)| String::from_utf8(key).unwrap())
            .collect();
        levels.push((len, keys));
    }
    assert_eq!(
        levels,
        vec![
            (1, vec!["\u{200C}a".to_string(), "\u{1F600}".to_string()]),
            (2, vec!["ab".to_string(), "a\u{200C}\u{301}b".to_string()]),
        ]
    );
    for term in terms {
        let expected = if term.starts_with('\u{1F600}') {
            1
        } else {
            term.chars().filter(|c| c.is_ascii()).count()
        };
        assert_eq!(zero_width.term_len(term.as_bytes()), expected, "{term:?}");
    }
    Ok(())
}

#[test]
fn test_search_shortest_first() -> crate::Result<()> {
    let terms: &[&str] = &["a", "a\"b", "ab", "abc", "ab\u{301}", "\u{e9}"];
    let buffer: Vec<u8> = {
        let mut builder = TermDictionaryBuilder::create(Vec::new())?;
        for (ord, term) in terms.iter().enumerate() {
            builder.insert(term.as_bytes(), &make_term_info(ord as u64))?;
        }
        builder.finish()?
    };
    let dict = TermDictionary::open(FileSlice::from(buffer))?;

    let levels = |zero_width: ZeroWidthChars| -> io::Result<Vec<(usize, Vec<String>)>> {
        let mut walk = dict.search_shortest_first(&tantivy_fst::automaton::AlwaysMatch, zero_width);
        let mut levels = Vec::new();
        while let Some((len, level)) = walk.next_level()? {
            let keys = level
                .into_iter()
                .map(|(key, info)| {
                    let term = String::from_utf8(key).unwrap();
                    let expected = terms.iter().position(|t| *t == term).unwrap();
                    assert_eq!(info, make_term_info(expected as u64), "term info of {term}");
                    term
                })
                .collect();
            levels.push((len, keys));
        }
        Ok(levels)
    };

    // every character counts: the quote makes a"b three characters long
    assert_eq!(
        levels(ZeroWidthChars::NONE)?,
        vec![
            (1, vec!["a".to_string(), "\u{e9}".to_string()]),
            (2, vec!["ab".to_string()]),
            (
                3,
                vec![
                    "a\"b".to_string(),
                    "abc".to_string(),
                    "ab\u{301}".to_string()
                ]
            ),
        ]
    );
    // a zero-width quote: a"b is as short as ab
    assert_eq!(
        levels(ZeroWidthChars::new(b"\""))?,
        vec![
            (1, vec!["a".to_string(), "\u{e9}".to_string()]),
            (2, vec!["a\"b".to_string(), "ab".to_string()]),
            (3, vec!["abc".to_string(), "ab\u{301}".to_string()]),
        ]
    );
    // a zero-width combining accent (a multi-byte character) too
    static COMBINING: [(char, char); 1] = [('\u{300}', '\u{36F}')];
    assert_eq!(
        levels(ZeroWidthChars::new(b"\"").with_ranges(&COMBINING))?,
        vec![
            (1, vec!["a".to_string(), "\u{e9}".to_string()]),
            (
                2,
                vec![
                    "a\"b".to_string(),
                    "ab".to_string(),
                    "ab\u{301}".to_string()
                ]
            ),
            (3, vec!["abc".to_string()]),
        ]
    );
    Ok(())
}
