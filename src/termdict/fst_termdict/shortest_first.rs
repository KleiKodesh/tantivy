//! Shortest-first automaton walk over the term FST.
//!
//! A normal automaton stream visits matches in byte (alphabetical) order, so
//! capping an expansion means walking all of it first. This walk goes
//! breadth-first by term length in characters: every match of length L is
//! produced before any of length L+1, and the caller stops pulling levels as
//! soon as its cap is reached. The long tail of a broad expansion (e.g.
//! `*כי*`) is then never visited: the walk cost is bounded by the cap, not
//! by the expansion.
//!
//! Length counts UTF-8 characters except `"` (the canonical intra-word quote
//! the Hebrew tokenizer keeps): a quote is zero-width, so רש"י and רשי sit on
//! the same level, as in FtsLib whose terms carried no quotes.

use tantivy_fst::raw::{CompiledAddr, Fst};
use tantivy_fst::Automaton;

use crate::termdict::TermOrdinal;
use crate::directory::OwnedBytes;

struct Frontier<S> {
    addr: CompiledAddr,
    ord: u64,
    state: S,
    key: Vec<u8>,
}

/// One level of a [`ShortestFirstWalk`]: every matching term of the same
/// length, as (key bytes, term ordinal), in byte order.
pub type Level = Vec<(Vec<u8>, TermOrdinal)>;

/// Iterates the automaton's matches level by level, shortest first.
pub struct ShortestFirstWalk<'a, A: Automaton> {
    fst: &'a Fst<OwnedBytes>,
    automaton: &'a A,
    frontier: Vec<Frontier<A::State>>,
    level: usize,
}

impl<'a, A> ShortestFirstWalk<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    pub(crate) fn new(fst: &'a Fst<OwnedBytes>, automaton: &'a A) -> Self {
        let start = automaton.start();
        let frontier = if automaton.can_match(&start) {
            vec![Frontier {
                addr: fst.root().addr(),
                ord: 0,
                state: start,
                key: Vec::new(),
            }]
        } else {
            Vec::new()
        };
        ShortestFirstWalk {
            fst,
            automaton,
            frontier,
            level: 0,
        }
    }

    /// The length (in characters) of the level the next call returns.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Follows one whole UTF-8 character (lead + continuation bytes) from
    /// `entry`, pushing every reachable node after it into `next`. Zero-width
    /// quote transitions go to `same` (this level) instead.
    fn expand(&self, entry: &Frontier<A::State>, same: &mut Vec<Frontier<A::State>>, next: &mut Vec<Frontier<A::State>>) {
        let node = self.fst.node(entry.addr);
        for t in node.transitions() {
            let state = self.automaton.accept(&entry.state, t.inp);
            if !self.automaton.can_match(&state) {
                continue;
            }
            let mut key = entry.key.clone();
            key.push(t.inp);
            let child = Frontier {
                addr: t.addr,
                ord: entry.ord + t.out.value(),
                state,
                key,
            };
            if t.inp == b'"' {
                same.push(child);
                continue;
            }
            let extra = match t.inp {
                0x00..=0x7F => 0,
                0xC0..=0xDF => 1,
                0xE0..=0xEF => 2,
                0xF0..=0xF7 => 3,
                _ => 0, // stray continuation byte: treat as its own char
            };
            self.complete_char(child, extra, next);
        }
    }

    /// Consumes `remaining` continuation bytes, then pushes into `next`.
    fn complete_char(&self, entry: Frontier<A::State>, remaining: usize, next: &mut Vec<Frontier<A::State>>) {
        if remaining == 0 {
            next.push(entry);
            return;
        }
        let node = self.fst.node(entry.addr);
        for t in node.transitions() {
            if t.inp & 0xC0 != 0x80 {
                continue;
            }
            let state = self.automaton.accept(&entry.state, t.inp);
            if !self.automaton.can_match(&state) {
                continue;
            }
            let mut key = entry.key.clone();
            key.push(t.inp);
            self.complete_char(
                Frontier {
                    addr: t.addr,
                    ord: entry.ord + t.out.value(),
                    state,
                    key,
                },
                remaining - 1,
                next,
            );
        }
    }
}

impl<A> Iterator for ShortestFirstWalk<'_, A>
where
    A: Automaton,
    A::State: Clone,
{
    type Item = Level;

    /// The next non-empty level of matches, or None when the walk is done.
    fn next(&mut self) -> Option<Level> {
        while !self.frontier.is_empty() {
            let mut queue = std::mem::take(&mut self.frontier);
            let mut next = Vec::new();
            let mut matches = Vec::new();
            while let Some(entry) = queue.pop() {
                let node = self.fst.node(entry.addr);
                if node.is_final() && self.automaton.is_match(&entry.state) {
                    matches.push((entry.key.clone(), entry.ord + node.final_output().value()));
                }
                self.expand(&entry, &mut queue, &mut next);
            }
            self.frontier = next;
            self.level += 1;
            if !matches.is_empty() {
                matches.sort_unstable();
                matches.dedup_by(|a, b| a.0 == b.0);
                return Some(matches);
            }
        }
        None
    }
}
