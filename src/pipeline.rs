//! Data generation pipeline: dictionary.csv + Gutenberg corpus → words.txt + .dat shards.
//!
//! Two-pass architecture:
//!   Pass 1 (fast): cheap tokenization, count raw adjacent word pairs where both
//!                  words are in the dictionary. Store example contexts.
//!   Pass 2 (targeted): POS-tag only the stored example sentences for top-frequency
//!                      pairs to classify them into pattern types.
//!
//! Feature-gated under `tagger`. This is exec-only code, not verified by Verus.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::pos::{Tagger, POS};

// ---------------------------------------------------------------------------
// Pattern code (FORMAT.md single-character codes)
// ---------------------------------------------------------------------------

/// Single-character pattern codes from FORMAT.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCode {
    // Noun headword (existing)
    AdjNoun,    // 'a' — ADJ NOUN, headword=noun(w1), collocate=adj(w0)
    VerbNoun,   // 'v' — VERB NOUN, headword=noun(w1), collocate=verb(w0)
    NounVerb,   // 'V' — NOUN VERB, headword=noun(w0), collocate=verb(w1)
    PrepNoun,   // 'n' — PREP NOUN, headword=noun(w1), collocate=prep(w0)
    NounNoun,   // 'N' — NOUN NOUN, headword=noun(w0), collocate=noun(w1)
    // Verb headword (new)
    VerbObject, // 'o' — VERB NOUN, headword=verb(w0), collocate=noun(w1)
    SubjVerb,   // 's' — NOUN VERB, headword=verb(w1), collocate=noun(w0)
    AdvVerb,    // 'd' — ADV VERB, headword=verb(w1), collocate=adv(w0)
    VerbAdv,    // 'D' — VERB ADV, headword=verb(w0), collocate=adv(w1)
    // Adjective headword (new)
    AdjObject,  // 'j' — ADJ NOUN, headword=adj(w0), collocate=noun(w1)
    AdvAdj,     // 'e' — ADV ADJ, headword=adj(w1), collocate=adv(w0)
}

impl PatternCode {
    pub fn code_char(self) -> char {
        match self {
            PatternCode::AdjNoun => 'a',
            PatternCode::VerbNoun => 'v',
            PatternCode::NounVerb => 'V',
            PatternCode::PrepNoun => 'n',
            PatternCode::NounNoun => 'N',
            PatternCode::VerbObject => 'o',
            PatternCode::SubjVerb => 's',
            PatternCode::AdvVerb => 'd',
            PatternCode::VerbAdv => 'D',
            PatternCode::AdjObject => 'j',
            PatternCode::AdvAdj => 'e',
        }
    }

    /// Canonical ordering for output (a, v, V, n, N, o, s, d, D, j, e).
    pub fn order(self) -> u8 {
        match self {
            PatternCode::AdjNoun => 0,
            PatternCode::VerbNoun => 1,
            PatternCode::NounVerb => 2,
            PatternCode::PrepNoun => 3,
            PatternCode::NounNoun => 4,
            PatternCode::VerbObject => 5,
            PatternCode::SubjVerb => 6,
            PatternCode::AdvVerb => 7,
            PatternCode::VerbAdv => 8,
            PatternCode::AdjObject => 9,
            PatternCode::AdvAdj => 10,
        }
    }

    /// The POS of the headword for this pattern.
    pub fn headword_pos(self) -> DictPOS {
        match self {
            PatternCode::AdjNoun
            | PatternCode::VerbNoun
            | PatternCode::PrepNoun
            | PatternCode::NounNoun
            | PatternCode::NounVerb => DictPOS::Noun,
            PatternCode::VerbObject
            | PatternCode::SubjVerb
            | PatternCode::AdvVerb
            | PatternCode::VerbAdv => DictPOS::Verb,
            PatternCode::AdjObject
            | PatternCode::AdvAdj => DictPOS::Adj,
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format seconds as human-readable duration (e.g. "2m 30s", "1h 05m").
fn fmt_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

// ---------------------------------------------------------------------------
// Base36 encoding
// ---------------------------------------------------------------------------

const BASE36_CHARS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Encode a non-negative integer as a base36 string.
pub fn base36(mut n: usize) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while n > 0 {
        digits.push(BASE36_CHARS[n % 36] as char);
        n /= 36;
    }
    digits.reverse();
    digits.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Dictionary parsing
// ---------------------------------------------------------------------------

/// Coarse POS category derived from dictionary wordtype strings.
/// Used to skip spaCy for unambiguous words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DictPOS {
    Adj,
    Noun,
    Verb,
    Adv,
    Prep,
}

/// Map a dictionary wordtype string (e.g. "n.", "v. t.", "a.") to a set of DictPOS.
/// Returns multiple for compound types like "a. & n." or "imp. & p. p." (verb form).
fn map_wordtype(wt: &str) -> Vec<DictPOS> {
    let wt = wt.trim().trim_end_matches('.');
    match wt {
        "n" | "n. pl" | "pl" => vec![DictPOS::Noun],
        "a" | "superl" => vec![DictPOS::Adj],
        "v" | "v. t" | "v. i" => vec![DictPOS::Verb],
        "adv" => vec![DictPOS::Adv],
        "prep" | "conj" => vec![DictPOS::Prep],
        // Compound types
        "v. t. & i" => vec![DictPOS::Verb],
        "a. & n" => vec![DictPOS::Adj, DictPOS::Noun],
        "n. & v" => vec![DictPOS::Noun, DictPOS::Verb],
        "a. & adv" => vec![DictPOS::Adj, DictPOS::Adv],
        // Participle / past participle forms — these function as verbs and adjectives
        "imp. & p. p" | "p. p" | "imp" => vec![DictPOS::Verb, DictPOS::Adj],
        "p. pr. & vb. n" => vec![DictPOS::Verb, DictPOS::Noun],
        "p. p. & a" => vec![DictPOS::Adj, DictPOS::Verb],
        _ => vec![],
    }
}

/// Result of parsing the dictionary.
pub struct Dictionary {
    /// Sorted word list (line number = word ID).
    pub words: Vec<String>,
    /// word → POS → all definitions for that POS.
    pub definitions: HashMap<String, HashMap<DictPOS, Vec<String>>>,
    /// word → set of possible POS categories from the dictionary.
    pub pos_sets: HashMap<String, HashSet<DictPOS>>,
}

/// Parse dictionary.csv → Dictionary with word list, definitions, and POS sets.
///
/// Words are lowercased and deduplicated. If a word has multiple definitions,
/// the first one is kept (shortest/simplest sense).
pub fn parse_dictionary(path: &Path) -> Dictionary {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .expect("failed to open dictionary.csv");

    let mut definitions: HashMap<String, HashMap<DictPOS, Vec<String>>> = HashMap::new();
    let mut pos_sets: HashMap<String, HashSet<DictPOS>> = HashMap::new();
    // Track which words exist (for word list generation).
    let mut all_words: HashSet<String> = HashSet::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        let word = record.get(0).unwrap_or("").trim().to_lowercase();
        if word.is_empty() || !word.chars().all(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        let definition = record.get(2).unwrap_or("").trim().to_string();
        let wordtype = record.get(1).unwrap_or("").trim();

        all_words.insert(word.clone());
        let pos_list = map_wordtype(wordtype);
        if !pos_list.is_empty() {
            let set = pos_sets.entry(word.clone()).or_default();
            let defs = definitions.entry(word).or_default();
            for pos in &pos_list {
                set.insert(*pos);
                if !definition.is_empty() {
                    let pos_defs = defs.entry(*pos).or_default();
                    if !pos_defs.contains(&definition) {
                        pos_defs.push(definition.clone());
                    }
                }
            }
        } else {
            // No POS mapped — store definition under a fallback
            // (will be used if we can't determine POS)
            let defs = definitions.entry(word).or_default();
            if !definition.is_empty() {
                let pos_defs = defs.entry(DictPOS::Noun).or_default();
                if !pos_defs.contains(&definition) {
                    pos_defs.push(definition.clone());
                }
            }
        }
    }

    let mut words: Vec<String> = all_words.into_iter().collect();
    words.sort();

    let unambiguous = pos_sets.values().filter(|s| s.len() == 1).count();
    let ambiguous = pos_sets.values().filter(|s| s.len() > 1).count();
    let no_pos = words.len() - pos_sets.len();
    eprintln!(
        "Dictionary POS: {} unambiguous, {} ambiguous, {} no POS tag",
        unambiguous, ambiguous, no_pos
    );

    Dictionary {
        words,
        definitions,
        pos_sets,
    }
}

/// Try to classify a bigram using dictionary POS alone (no spaCy needed).
/// Returns None if either word is ambiguous or missing POS info.
/// Returns Vec of PatternCodes — symmetric bigrams emit BOTH directions.
fn classify_from_dict(
    w0: &str,
    w1: &str,
    pos_sets: &HashMap<String, HashSet<DictPOS>>,
) -> Option<Vec<PatternCode>> {
    let s0 = pos_sets.get(w0)?;
    let s1 = pos_sets.get(w1)?;

    // Only use dict POS if both words are unambiguous
    if s0.len() != 1 || s1.len() != 1 {
        return None;
    }

    let p0 = *s0.iter().next().unwrap();
    let p1 = *s1.iter().next().unwrap();

    let mut patterns = Vec::new();
    match (p0, p1) {
        (DictPOS::Adj, DictPOS::Noun) => {
            patterns.push(PatternCode::AdjNoun);   // noun gets 'a'
            patterns.push(PatternCode::AdjObject);  // adj gets 'j'
        }
        (DictPOS::Verb, DictPOS::Noun) => {
            patterns.push(PatternCode::VerbNoun);   // noun gets 'v'
            patterns.push(PatternCode::VerbObject); // verb gets 'o'
        }
        (DictPOS::Noun, DictPOS::Verb) => {
            patterns.push(PatternCode::NounVerb);   // noun gets 'V'
            patterns.push(PatternCode::SubjVerb);   // verb gets 's'
        }
        (DictPOS::Prep, DictPOS::Noun) => {
            patterns.push(PatternCode::PrepNoun);   // noun gets 'n' (no other side)
        }
        (DictPOS::Noun, DictPOS::Noun) => {
            patterns.push(PatternCode::NounNoun);
            patterns.push(PatternCode::PrepNoun);
        }
        (DictPOS::Adv, DictPOS::Verb) => {
            patterns.push(PatternCode::AdvVerb);    // verb gets 'd'
        }
        (DictPOS::Verb, DictPOS::Adv) => {
            patterns.push(PatternCode::VerbAdv);    // verb gets 'D'
        }
        (DictPOS::Adv, DictPOS::Adj) => {
            patterns.push(PatternCode::AdvAdj);     // adj gets 'e'
        }
        _ => return Some(vec![]), // Known POS but not a collocation pattern
    }
    Some(patterns)
}

/// Check if any combination of POS from two sets could produce a valid pattern.
/// Returns false if we can prove no pattern is possible (skip spaCy entirely).
fn any_pattern_possible(
    w0: &str,
    w1: &str,
    pos_sets: &HashMap<String, HashSet<DictPOS>>,
) -> bool {
    // If a word has no dict POS at all, we can't rule anything out
    let s0 = match pos_sets.get(w0) {
        Some(s) => s,
        None => return true,
    };
    let s1 = match pos_sets.get(w1) {
        Some(s) => s,
        None => return true,
    };

    // Check all combos: does any (p0, p1) pair form a valid pattern?
    for &p0 in s0 {
        for &p1 in s1 {
            match (p0, p1) {
                (DictPOS::Adj, DictPOS::Noun)
                | (DictPOS::Verb, DictPOS::Noun)
                | (DictPOS::Noun, DictPOS::Verb)
                | (DictPOS::Prep, DictPOS::Noun)
                | (DictPOS::Noun, DictPOS::Noun)
                | (DictPOS::Adv, DictPOS::Verb)
                | (DictPOS::Verb, DictPOS::Adv)
                | (DictPOS::Adv, DictPOS::Adj) => return true,
                _ => {}
            }
        }
    }
    false
}

/// Classify a bigram using tagger-resolved POS (from the tagger's known-word
/// lookup table). Returns patterns if both POS values map to a valid combo.
fn classify_from_tagger_pos(p0: POS, p1: POS) -> Vec<PatternCode> {
    match (p0, p1) {
        (POS::Adj, POS::Noun) => vec![PatternCode::AdjNoun, PatternCode::AdjObject],
        (POS::Verb, POS::Noun) => vec![PatternCode::VerbNoun, PatternCode::VerbObject],
        (POS::Noun, POS::Verb) => vec![PatternCode::NounVerb, PatternCode::SubjVerb],
        (POS::Prep, POS::Noun) => vec![PatternCode::PrepNoun],
        (POS::Noun, POS::Noun) => vec![PatternCode::NounNoun, PatternCode::PrepNoun],
        (POS::Adv, POS::Verb) => vec![PatternCode::AdvVerb],
        (POS::Verb, POS::Adv) => vec![PatternCode::VerbAdv],
        (POS::Adv, POS::Adj) => vec![PatternCode::AdvAdj],
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Cheap tokenizer for Pass 1
// ---------------------------------------------------------------------------

/// Strip leading/trailing punctuation and lowercase a token.
/// Returns None if the result is empty or not all-alphabetic.
fn normalize_token(raw: &str) -> Option<String> {
    let trimmed = raw.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if trimmed.is_empty() {
        return None;
    }
    // Fast reject before allocating: all bytes must be ASCII alphabetic
    if !trimmed.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

/// Normalize a token and look up its interned ID without heap allocation.
/// Uses a stack buffer for lowercasing to avoid allocating a String
/// for the vast majority of tokens that aren't in the dictionary.
fn normalize_and_intern(raw: &str, interner: &WordInterner) -> Option<u32> {
    let trimmed = raw.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    if trimmed.len() <= 64 {
        let mut buf = [0u8; 64];
        for (i, b) in trimmed.bytes().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        // SAFETY: input was all ASCII alphabetic, lowercase is still valid UTF-8
        let lower = unsafe { std::str::from_utf8_unchecked(&buf[..trimmed.len()]) };
        interner.get(lower)
    } else {
        let lower = trimmed.to_ascii_lowercase();
        interner.get(&lower)
    }
}

/// Normalize a token and check stopword membership without heap allocation.
/// Returns the lowercased String only if it passes the stopword filter.
fn normalize_filtered(raw: &str, stopwords: &HashSet<String>) -> Option<String> {
    let trimmed = raw.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    // Check stopword on stack buffer before allocating
    if trimmed.len() <= 64 {
        let mut buf = [0u8; 64];
        for (i, b) in trimmed.bytes().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        let lower = unsafe { std::str::from_utf8_unchecked(&buf[..trimmed.len()]) };
        if stopwords.contains(lower) {
            return None;
        }
        Some(lower.to_owned())
    } else {
        let lower = trimmed.to_ascii_lowercase();
        if stopwords.contains(&lower) {
            None
        } else {
            Some(lower)
        }
    }
}

// ---------------------------------------------------------------------------
// Corpus data types
// ---------------------------------------------------------------------------

/// Metadata from a Gutenberg JSON line.
#[derive(serde::Deserialize)]
struct GutenbergRecord {
    text: String,
    metadata: GutenbergMeta,
}

#[derive(serde::Deserialize)]
struct GutenbergMeta {
    title: Option<String>,
}

// ---------------------------------------------------------------------------
// Pass 1: Raw bigram counting (parallel, interned)
// ---------------------------------------------------------------------------

/// Result of Pass 1: bigram counts and unigram counts (no examples).
pub struct Pass1Result {
    /// (word_a, word_b) → raw count (word_a appears before word_b).
    pub bigram_counts: HashMap<(String, String), u64>,
    /// word → total occurrence count (as part of any non-stopword bigram).
    pub unigram_counts: HashMap<String, u64>,
    /// Total number of bigrams counted (for PMI normalization).
    pub total_bigrams: u64,
    /// Total number of books processed.
    pub total_books: usize,
}

/// Load stopwords from a file (one word per line). Returns empty set if file missing.
pub fn load_stopwords(path: &Path) -> HashSet<String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => contents
            .lines()
            .map(|l| l.trim().to_lowercase())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => HashSet::new(),
    }
}

/// Word interner: maps words → u32 IDs for compact HashMap keys.
struct WordInterner {
    /// word string → ID (FxHashMap for fast string lookups)
    word_to_id: FxHashMap<String, u32>,
    /// ID → word string
    id_to_word: Vec<String>,
}

impl WordInterner {
    /// Build interner from the dictionary word set (only these words matter).
    fn from_dict(dict_words: &HashMap<String, usize>, stopwords: &HashSet<String>) -> Self {
        let mut word_to_id = FxHashMap::default();
        let mut id_to_word = Vec::new();
        for word in dict_words.keys() {
            if !stopwords.contains(word.as_str()) {
                let id = id_to_word.len() as u32;
                word_to_id.insert(word.clone(), id);
                id_to_word.push(word.clone());
            }
        }
        WordInterner { word_to_id, id_to_word }
    }

    fn get(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(word).copied()
    }

    fn word(&self, id: u32) -> &str {
        &self.id_to_word[id as usize]
    }
}

/// Per-book accumulator using interned (u32, u32) keys — counts only, no examples.
struct BookCounts {
    bigram_counts: FxHashMap<(u32, u32), u64>,
    /// Sparse unigram counts (good for small books with few unique words).
    unigram_counts: FxHashMap<u32, u64>,
    total_bigrams: u64,
}

/// Process a single book's text for pass 1: count bigrams only (no examples).
///
/// Uses a sliding window (no per-sentence Vec allocation) and zero-alloc
/// token normalization. Sentence boundaries (.!?) reset the window.
fn process_book_counts(
    text: &str,
    interner: &WordInterner,
) -> BookCounts {
    let mut bigram_counts: FxHashMap<(u32, u32), u64> = FxHashMap::default();
    let mut unigram_counts: FxHashMap<u32, u64> = FxHashMap::default();
    let mut total_bigrams: u64 = 0;

    let mut prev: Option<u32> = None;
    for raw in text.split_whitespace() {
        // Reset on sentence boundary
        if raw.bytes().any(|b| b == b'.' || b == b'!' || b == b'?') {
            prev = None;
            continue;
        }
        let cur = normalize_and_intern(raw, interner);
        if let (Some(id0), Some(id1)) = (prev, cur) {
            *bigram_counts.entry((id0, id1)).or_insert(0) += 1;
            *unigram_counts.entry(id0).or_insert(0) += 1;
            *unigram_counts.entry(id1).or_insert(0) += 1;
            total_bigrams += 1;
        }
        prev = cur;
    }

    BookCounts { bigram_counts, unigram_counts, total_bigrams }
}

/// Extract context-window sentences from a book for target bigram pairs (pass 2).
fn extract_sentences(
    text: &str,
    interner: &WordInterner,
    target_pairs: &HashSet<(u32, u32)>,
) -> Vec<String> {
    let mut sentences = Vec::new();
    for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
        let raw_words: Vec<&str> = sentence.split_whitespace().collect();
        let token_ids: Vec<Option<u32>> = raw_words
            .iter()
            .map(|w| normalize_and_intern(w, interner))
            .collect();

        for i in 0..token_ids.len().saturating_sub(1) {
            if let (Some(id0), Some(id1)) = (token_ids[i], token_ids[i + 1]) {
                if target_pairs.contains(&(id0, id1)) {
                    let start = i.saturating_sub(2);
                    let end = (i + 4).min(raw_words.len());
                    sentences.push(raw_words[start..end].join(" "));
                }
            }
        }
    }
    sentences
}

/// Read up to `batch_size` JSON records from a buffered reader.
/// Returns an empty Vec at EOF.
fn read_batch(reader: &mut impl BufRead, batch_size: usize) -> Vec<GutenbergRecord> {
    let mut batch = Vec::with_capacity(batch_size);
    let mut line_buf = String::new();
    while batch.len() < batch_size {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("warning: skipping line due to read error: {}", e);
                continue;
            }
        }
        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str(trimmed) {
            Ok(r) => batch.push(r),
            Err(e) => eprintln!("warning: skipping malformed JSON: {}", e),
        }
    }
    batch
}

const BATCH_SIZE: usize = 1000;

/// Pre-scan corpus to count unigram word frequencies (for `--all-words`).
///
/// Streams corpus in batches, counts normalized words (excluding stopwords)
/// using `par_iter`, merges into a running `HashMap<String, u64>`.
pub fn prescan_word_frequencies(
    corpus_paths: &[PathBuf],
    stopwords: &HashSet<String>,
    max_books: Option<usize>,
) -> HashMap<String, u64> {
    let mut freq: HashMap<String, u64> = HashMap::new();
    let mut total_books: usize = 0;
    let start = std::time::Instant::now();

    for corpus_path in corpus_paths {
        eprintln!("Pre-scan: reading {}...", corpus_path.display());
        let file = File::open(corpus_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", corpus_path.display(), e));
        let decoder = flate2::read::GzDecoder::new(file);
        let mut reader = BufReader::new(decoder);

        loop {
            let remaining = max_books.map(|m| m.saturating_sub(total_books));
            if remaining == Some(0) {
                break;
            }
            let this_batch_size = remaining.map(|r| r.min(BATCH_SIZE)).unwrap_or(BATCH_SIZE);

            let batch = read_batch(&mut reader, this_batch_size);
            if batch.is_empty() {
                break;
            }
            let batch_len = batch.len();

            // Count words per book in parallel
            let book_freqs: Vec<HashMap<String, u64>> = batch
                .par_iter()
                .map(|record| {
                    let mut local: HashMap<String, u64> = HashMap::new();
                    for word in record.text.split_whitespace() {
                        if let Some(norm) = normalize_token(word) {
                            if !stopwords.contains(&norm) {
                                *local.entry(norm).or_insert(0) += 1;
                            }
                        }
                    }
                    local
                })
                .collect();

            // Merge into running totals
            for bf in book_freqs {
                for (word, count) in bf {
                    *freq.entry(word).or_insert(0) += count;
                }
            }

            total_books += batch_len;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_books as f64 / elapsed;
            eprint!(
                "\r  Pre-scan: {} books | {:.0} books/s | {}\x1b[K",
                total_books, rate, fmt_duration(elapsed),
            );

            if remaining == Some(batch_len) {
                break;
            }
        }

        if max_books.map(|m| total_books >= m).unwrap_or(false) {
            break;
        }
    }
    eprintln!();
    eprintln!(
        "Pre-scan: {} unique words from {} books",
        freq.len(),
        total_books,
    );
    freq
}

/// Pass 1: streaming parallel corpus processing with interned word IDs.
///
/// Reads corpus in batches of 1000 books, processes each batch in parallel,
/// merges into running totals, then drops the batch to free memory.
pub fn pass1_count_bigrams(
    corpus_paths: &[PathBuf],
    dict_words: &HashMap<String, usize>,
    stopwords: &HashSet<String>,
    max_books: Option<usize>,
) -> Pass1Result {
    let interner = WordInterner::from_dict(dict_words, stopwords);
    let num_words = interner.id_to_word.len();
    eprintln!("Interner: {} words indexed", num_words);

    let mut bigram_counts: FxHashMap<(u32, u32), u64> = FxHashMap::default();
    let mut unigram_counts: Vec<u64> = vec![0u64; num_words];
    let mut total_bigrams: u64 = 0;
    let mut total_books: usize = 0;
    let start = std::time::Instant::now();

    for corpus_path in corpus_paths {
        eprintln!("Reading {}...", corpus_path.display());
        let file = File::open(corpus_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", corpus_path.display(), e));
        let decoder = flate2::read::GzDecoder::new(file);
        let mut reader = BufReader::new(decoder);

        loop {
            // Respect max_books across all files
            let remaining = max_books.map(|m| m.saturating_sub(total_books));
            if remaining == Some(0) {
                eprintln!("Reached --max-books limit ({})", max_books.unwrap());
                break;
            }
            let this_batch_size = remaining.map(|r| r.min(BATCH_SIZE)).unwrap_or(BATCH_SIZE);

            let batch = read_batch(&mut reader, this_batch_size);
            if batch.is_empty() {
                break;
            }
            let batch_len = batch.len();

            // Process batch in parallel with fold+reduce.
            // Per-book counts are sparse (small books), thread accumulators are dense.
            let (batch_bigrams, batch_unigrams, batch_total) = batch
                .par_iter()
                .map(|record| process_book_counts(&record.text, &interner))
                .fold(
                    || (FxHashMap::<(u32, u32), u64>::default(), vec![0u64; num_words], 0u64),
                    |(mut bg, mut ug, mut tot), bc| {
                        tot += bc.total_bigrams;
                        for (key, count) in bc.bigram_counts {
                            *bg.entry(key).or_insert(0) += count;
                        }
                        for (id, count) in bc.unigram_counts {
                            ug[id as usize] += count;
                        }
                        (bg, ug, tot)
                    },
                )
                .reduce(
                    || (FxHashMap::default(), vec![0u64; num_words], 0u64),
                    |(mut bg_a, mut ug_a, tot_a), (bg_b, ug_b, tot_b)| {
                        for (key, count) in bg_b {
                            *bg_a.entry(key).or_insert(0) += count;
                        }
                        for (i, count) in ug_b.into_iter().enumerate() {
                            ug_a[i] += count;
                        }
                        (bg_a, ug_a, tot_a + tot_b)
                    },
                );

            // Merge batch result into running totals
            total_bigrams += batch_total;
            for (key, count) in batch_bigrams {
                *bigram_counts.entry(key).or_insert(0) += count;
            }
            for (i, count) in batch_unigrams.into_iter().enumerate() {
                unigram_counts[i] += count;
            }

            total_books += batch_len;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_books as f64 / elapsed;
            eprint!(
                "\r  Pass 1: {} books | {:.0} books/s | {}\x1b[K",
                total_books, rate, fmt_duration(elapsed),
            );

            if remaining == Some(batch_len) {
                eprintln!("\nReached --max-books limit ({})", max_books.unwrap());
                break;
            }
        }

        if max_books.map(|m| total_books >= m).unwrap_or(false) {
            break;
        }
    }
    eprintln!(); // newline after progress

    // Convert interned IDs back to strings
    let str_bigram_counts: HashMap<(String, String), u64> = bigram_counts
        .into_iter()
        .map(|((a, b), c)| ((interner.word(a).to_string(), interner.word(b).to_string()), c))
        .collect();
    let str_unigram_counts: HashMap<String, u64> = unigram_counts
        .into_iter()
        .enumerate()
        .filter(|(_, c)| *c > 0)
        .map(|(id, c)| (interner.word(id as u32).to_string(), c))
        .collect();

    eprintln!(
        "Pass 1: {} books, {} unique bigrams, {} total bigrams, {} unique words",
        total_books,
        str_bigram_counts.len(),
        total_bigrams,
        str_unigram_counts.len(),
    );
    Pass1Result {
        bigram_counts: str_bigram_counts,
        unigram_counts: str_unigram_counts,
        total_bigrams,
        total_books,
    }
}

// ---------------------------------------------------------------------------
// Pass 1 (all-words mode): count ALL bigrams without interner filtering.
// Trades memory for one fewer corpus read when --all-words is used.
// ---------------------------------------------------------------------------

/// Per-book accumulator using String keys (no interner filtering).
struct BookCountsAll {
    bigrams: HashMap<(String, String), u64>,
    unigrams: HashMap<String, u64>,
    total_bigrams: u64,
}

/// Process a single book's text counting ALL non-stopword bigrams and unigrams.
/// Uses a per-book interner to avoid String cloning during counting, and sliding
/// window with sentence-boundary reset (no per-sentence Vec allocation).
fn process_book_all(text: &str, stopwords: &HashSet<String>) -> BookCountsAll {
    // Per-book interner: assign u32 IDs to words within this book
    let mut local_to_id: FxHashMap<String, u32> = FxHashMap::default();
    let mut local_words: Vec<String> = Vec::new();
    let mut bigrams: FxHashMap<(u32, u32), u64> = FxHashMap::default();
    let mut unigrams: Vec<u64> = Vec::new();
    let mut total_bigrams: u64 = 0;

    let mut prev: Option<u32> = None;
    for raw in text.split_whitespace() {
        if raw.bytes().any(|b| b == b'.' || b == b'!' || b == b'?') {
            prev = None;
            continue;
        }
        if let Some(cur_word) = normalize_filtered(raw, stopwords) {
            let cur_id = match local_to_id.get(&cur_word) {
                Some(&id) => id,
                None => {
                    let id = local_words.len() as u32;
                    local_to_id.insert(cur_word.clone(), id);
                    local_words.push(cur_word);
                    unigrams.push(0);
                    id
                }
            };
            unigrams[cur_id as usize] += 1;
            if let Some(prev_id) = prev {
                *bigrams.entry((prev_id, cur_id)).or_insert(0) += 1;
                total_bigrams += 1;
            }
            prev = Some(cur_id);
        } else {
            prev = None;
        }
    }

    // Convert back to String keys
    let str_bigrams: HashMap<(String, String), u64> = bigrams
        .into_iter()
        .map(|((a, b), c)| {
            ((local_words[a as usize].clone(), local_words[b as usize].clone()), c)
        })
        .collect();
    let str_unigrams: HashMap<String, u64> = unigrams
        .into_iter()
        .enumerate()
        .filter(|(_, c)| *c > 0)
        .map(|(i, c)| (local_words[i].clone(), c))
        .collect();

    BookCountsAll { bigrams: str_bigrams, unigrams: str_unigrams, total_bigrams }
}

/// Pass 1 (all-words mode): count ALL non-stopword bigrams and unigrams.
///
/// Unlike `pass1_count_bigrams`, this does NOT filter by dictionary words.
/// The caller should filter the results after expanding the dictionary.
/// This avoids a separate prescan step, saving one full corpus read.
pub fn pass1_count_all(
    corpus_paths: &[PathBuf],
    stopwords: &HashSet<String>,
    max_books: Option<usize>,
) -> Pass1Result {
    let mut bigram_counts: HashMap<(String, String), u64> = HashMap::new();
    let mut unigram_counts: HashMap<String, u64> = HashMap::new();
    let mut total_bigrams: u64 = 0;
    let mut total_books: usize = 0;
    let start = std::time::Instant::now();

    for corpus_path in corpus_paths {
        eprintln!("Reading {}...", corpus_path.display());
        let file = File::open(corpus_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", corpus_path.display(), e));
        let decoder = flate2::read::GzDecoder::new(file);
        let mut reader = BufReader::new(decoder);

        loop {
            let remaining = max_books.map(|m| m.saturating_sub(total_books));
            if remaining == Some(0) {
                eprintln!("Reached --max-books limit ({})", max_books.unwrap());
                break;
            }
            let this_batch_size = remaining.map(|r| r.min(BATCH_SIZE)).unwrap_or(BATCH_SIZE);

            let batch = read_batch(&mut reader, this_batch_size);
            if batch.is_empty() {
                break;
            }
            let batch_len = batch.len();

            // Process batch in parallel with fold+reduce
            let batch_merged = batch
                .par_iter()
                .map(|record| process_book_all(&record.text, stopwords))
                .fold(
                    || BookCountsAll {
                        bigrams: HashMap::new(),
                        unigrams: HashMap::new(),
                        total_bigrams: 0,
                    },
                    |mut acc, bc| {
                        acc.total_bigrams += bc.total_bigrams;
                        for (key, count) in bc.bigrams {
                            *acc.bigrams.entry(key).or_insert(0) += count;
                        }
                        for (word, count) in bc.unigrams {
                            *acc.unigrams.entry(word).or_insert(0) += count;
                        }
                        acc
                    },
                )
                .reduce(
                    || BookCountsAll {
                        bigrams: HashMap::new(),
                        unigrams: HashMap::new(),
                        total_bigrams: 0,
                    },
                    |mut a, b| {
                        a.total_bigrams += b.total_bigrams;
                        for (key, count) in b.bigrams {
                            *a.bigrams.entry(key).or_insert(0) += count;
                        }
                        for (word, count) in b.unigrams {
                            *a.unigrams.entry(word).or_insert(0) += count;
                        }
                        a
                    },
                );

            // Merge batch result into running totals
            total_bigrams += batch_merged.total_bigrams;
            for (key, count) in batch_merged.bigrams {
                *bigram_counts.entry(key).or_insert(0) += count;
            }
            for (word, count) in batch_merged.unigrams {
                *unigram_counts.entry(word).or_insert(0) += count;
            }

            total_books += batch_len;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total_books as f64 / elapsed;
            eprint!(
                "\r  Pass 1 (all): {} books | {:.0} books/s | {} | {} bigrams, {} words\x1b[K",
                total_books, rate, fmt_duration(elapsed),
                bigram_counts.len(), unigram_counts.len(),
            );

            if remaining == Some(batch_len) {
                eprintln!("\nReached --max-books limit ({})", max_books.unwrap());
                break;
            }
        }

        if max_books.map(|m| total_books >= m).unwrap_or(false) {
            break;
        }
    }
    eprintln!(); // newline after progress

    eprintln!(
        "Pass 1 (all): {} books, {} unique bigrams, {} total bigrams, {} unique words",
        total_books, bigram_counts.len(), total_bigrams, unigram_counts.len(),
    );
    Pass1Result {
        bigram_counts,
        unigram_counts,
        total_bigrams,
        total_books,
    }
}

// ---------------------------------------------------------------------------
// Pass 2: POS-tag candidates to classify patterns
// ---------------------------------------------------------------------------

/// Score entry: PMI score + raw frequency count.
#[derive(Debug, Clone, Copy)]
pub struct ScoreEntry {
    pub pmi: f64,
    pub count: u64,
}

/// Triple key: (headword, pattern, collocate) → scores.
pub type CollocationCounts = HashMap<(String, PatternCode, String), ScoreEntry>;

/// Compute PMI: log2(P(w0,w1) / (P(w0) * P(w1)))
///   = log2(bigram_count * total_bigrams / (unigram_w0 * unigram_w1))
fn pmi(bigram_count: u64, unigram_w0: u64, unigram_w1: u64, total: u64) -> f64 {
    if bigram_count == 0 || unigram_w0 == 0 || unigram_w1 == 0 || total == 0 {
        return 0.0;
    }
    ((bigram_count as f64) * (total as f64) / (unigram_w0 as f64 * unigram_w1 as f64)).log2()
}

/// Given a pattern and the bigram (w0, w1), return (headword, collocate).
fn headword_collocate<'a>(pat: &PatternCode, w0: &'a str, w1: &'a str) -> (&'a str, &'a str) {
    match pat {
        // Noun headword
        PatternCode::AdjNoun => (w1, w0),   // ADJ NOUN → noun is headword
        PatternCode::VerbNoun => (w1, w0),   // VERB NOUN → noun is headword
        PatternCode::NounVerb => (w0, w1),   // NOUN VERB → noun is headword
        PatternCode::PrepNoun => (w1, w0),   // PREP NOUN → noun is headword
        PatternCode::NounNoun => (w0, w1),   // NOUN NOUN → first noun is headword
        // Verb headword
        PatternCode::VerbObject => (w0, w1), // VERB NOUN → verb is headword
        PatternCode::SubjVerb => (w1, w0),   // NOUN VERB → verb is headword
        PatternCode::AdvVerb => (w1, w0),    // ADV VERB → verb is headword
        PatternCode::VerbAdv => (w0, w1),    // VERB ADV → verb is headword
        // Adjective headword
        PatternCode::AdjObject => (w0, w1),  // ADJ NOUN → adj is headword
        PatternCode::AdvAdj => (w1, w0),     // ADV ADJ → adj is headword
    }
}

/// Pass 2: for each top-frequency bigram, classify its pattern.
///
/// Fast path: if both words have unambiguous dictionary POS, classify directly.
/// Slow path: re-read corpus in batches, extract sentences for ambiguous pairs,
///            POS-tag them, and vote on pattern classification.
///
/// `top_per_word`: only classify the top N bigrams per word (by count).
pub fn pass2_classify(
    pass1: &Pass1Result,
    corpus_paths: &[PathBuf],
    dict_words: &HashMap<String, usize>,
    stopwords: &HashSet<String>,
    tagger: &dyn Tagger,
    pos_sets: &HashMap<String, HashSet<DictPOS>>,
    min_count: u64,
    top_per_word: usize,
    max_books: Option<usize>,
    _max_examples: usize,
) -> CollocationCounts {
    // Group bigrams by each participating word, keep top N per word.
    let mut per_word: HashMap<&str, Vec<(&(String, String), u64)>> = HashMap::new();
    for (pair, &count) in &pass1.bigram_counts {
        if count < min_count {
            continue;
        }
        per_word
            .entry(pair.0.as_str())
            .or_default()
            .push((pair, count));
        per_word
            .entry(pair.1.as_str())
            .or_default()
            .push((pair, count));
    }

    // For each word, sort by count descending and keep top_per_word.
    // Collect the union of selected bigrams.
    let mut selected: HashMap<&(String, String), u64> = HashMap::new();
    for entries in per_word.values_mut() {
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(top_per_word);
        for &(pair, count) in entries.iter() {
            selected.insert(pair, count);
        }
    }

    // Pre-classify: resolve dict-only pairs upfront, then try tagger lookup,
    // and collect remaining pairs that need full tagging.
    let mut counts: CollocationCounts = HashMap::new();
    let mut spacy_needed: Vec<(&(String, String), u64)> = Vec::new();
    let mut dict_classified = 0usize;
    let mut tagger_classified = 0usize;
    let mut dict_skipped = 0usize;

    for (&pair, &raw_count) in &selected {
        let (w0, w1) = pair;
        match classify_from_dict(w0, w1, pos_sets) {
            Some(pats) if pats.is_empty() => {
                // Known POS but not a collocation pattern (e.g. ADV+ADV)
                dict_skipped += 1;
            }
            Some(pats) => {
                // Both unambiguous, classified without tagger
                dict_classified += 1;
                let uw0 = pass1.unigram_counts.get(w0).copied().unwrap_or(0);
                let uw1 = pass1.unigram_counts.get(w1).copied().unwrap_or(0);
                let score = pmi(raw_count, uw0, uw1, pass1.total_bigrams);
                for pat in &pats {
                    let (headword, collocate) = headword_collocate(pat, w0, w1);
                    let entry = counts
                        .entry((headword.to_string(), *pat, collocate.to_string()))
                        .or_insert(ScoreEntry { pmi: 0.0, count: 0 });
                    if score > entry.pmi {
                        entry.pmi = score;
                        entry.count = raw_count;
                    }
                }
            }
            None => {
                // Try tagger's known-word lookup before falling through to full tagging
                if let (Some(p0), Some(p1)) = (tagger.lookup_pos(w0), tagger.lookup_pos(w1)) {
                    let pats = classify_from_tagger_pos(p0, p1);
                    if pats.is_empty() {
                        dict_skipped += 1;
                    } else {
                        tagger_classified += 1;
                        let uw0 = pass1.unigram_counts.get(w0).copied().unwrap_or(0);
                        let uw1 = pass1.unigram_counts.get(w1).copied().unwrap_or(0);
                        let score = pmi(raw_count, uw0, uw1, pass1.total_bigrams);
                        for pat in &pats {
                            let (headword, collocate) = headword_collocate(pat, w0, w1);
                            let entry = counts
                                .entry((headword.to_string(), *pat, collocate.to_string()))
                                .or_insert(ScoreEntry { pmi: 0.0, count: 0 });
                            if score > entry.pmi {
                                entry.pmi = score;
                                entry.count = raw_count;
                            }
                        }
                    }
                } else if any_pattern_possible(w0, w1, pos_sets) {
                    spacy_needed.push((pair, raw_count));
                } else {
                    dict_skipped += 1;
                }
            }
        }
    }
    spacy_needed.sort_by(|a, b| b.1.cmp(&a.1));

    eprintln!(
        "Pass 2: {} total — {} dict, {} tagger-lookup, {} skipped, {} need tagging",
        selected.len(), dict_classified, tagger_classified, dict_skipped, spacy_needed.len()
    );

    // Dump spaCy-needed pairs with reason for debugging
    {
        let dump_path = "spacy_needed.txt";
        let mut f = BufWriter::new(File::create(dump_path).expect("failed to create spacy_needed.txt"));
        writeln!(f, "# bigram\tcount\treason\tw0_pos\tw1_pos").unwrap();
        for (pair, count) in &spacy_needed {
            let (w0, w1) = pair;
            let s0 = pos_sets.get(w0.as_str());
            let s1 = pos_sets.get(w1.as_str());
            let reason = match (s0, s1) {
                (None, None) => "both_missing",
                (None, _) => "w0_missing",
                (_, None) => "w1_missing",
                (Some(a), Some(b)) if a.len() > 1 && b.len() > 1 => "both_ambiguous",
                (Some(a), _) if a.len() > 1 => "w0_ambiguous",
                (_, Some(b)) if b.len() > 1 => "w1_ambiguous",
                _ => "unknown",
            };
            let fmt_pos = |s: Option<&HashSet<DictPOS>>| match s {
                None => "NONE".to_string(),
                Some(set) => format!("{:?}", set),
            };
            writeln!(f, "{} {}\t{}\t{}\t{}\t{}", w0, w1, count, reason, fmt_pos(s0), fmt_pos(s1)).unwrap();
        }
        eprintln!("Wrote {} to {}", spacy_needed.len(), dump_path);
    }

    // Build interned target_pairs set for fast lookup during sentence extraction
    let interner = WordInterner::from_dict(dict_words, stopwords);
    let spacy_pair_set: HashSet<(String, String)> =
        spacy_needed.iter().map(|(pair, _)| (pair.0.clone(), pair.1.clone())).collect();
    let target_pairs: HashSet<(u32, u32)> = spacy_pair_set
        .iter()
        .filter_map(|(w0, w1)| {
            let id0 = interner.get(w0)?;
            let id1 = interner.get(w1)?;
            Some((id0, id1))
        })
        .collect();

    eprintln!(
        "  {} target pairs for streaming sentence extraction",
        target_pairs.len()
    );

    // Stream corpus again in batches, extract sentences, tag, vote
    let mut pair_votes: HashMap<(String, String), HashMap<PatternCode, u32>> = HashMap::new();
    let mut total_books: usize = 0;
    let mut total_sentences_tagged: usize = 0;
    let mut total_sentences_extracted: usize = 0;
    let tag_start = std::time::Instant::now();
    let spacy_batch_size = 100_000;

    let total = spacy_needed.len();

    for corpus_path in corpus_paths {
        let file = File::open(corpus_path)
            .unwrap_or_else(|e| panic!("failed to open {}: {}", corpus_path.display(), e));
        let decoder = flate2::read::GzDecoder::new(file);
        let mut reader = BufReader::new(decoder);

        loop {
            let remaining = max_books.map(|m| m.saturating_sub(total_books));
            if remaining == Some(0) {
                break;
            }
            let this_batch_size = remaining.map(|r| r.min(BATCH_SIZE)).unwrap_or(BATCH_SIZE);

            let batch = read_batch(&mut reader, this_batch_size);
            if batch.is_empty() {
                break;
            }
            let batch_len = batch.len();

            // Extract sentences in parallel for target pairs only
            let batch_sentences: Vec<Vec<String>> = batch
                .par_iter()
                .map(|record| extract_sentences(&record.text, &interner, &target_pairs))
                .collect();
            drop(batch); // free book texts

            // Flatten sentences from all books in this batch
            let all_sentences: Vec<String> = batch_sentences.into_iter().flatten().collect();
            total_sentences_extracted += all_sentences.len();

            // Tag the batch immediately
            if !all_sentences.is_empty() {
                for chunk_start in (0..all_sentences.len()).step_by(spacy_batch_size) {
                    let chunk_end = (chunk_start + spacy_batch_size).min(all_sentences.len());
                    let text_refs: Vec<&str> = all_sentences[chunk_start..chunk_end]
                        .iter()
                        .map(|s| s.as_str())
                        .collect();
                    let tagged_batch = match tagger.tag_batch(&text_refs, spacy_batch_size) {
                        Ok(results) => results,
                        Err(e) => {
                            let msg = e.to_string();
                            if msg.contains("KeyboardInterrupt") {
                                eprintln!("\n  Interrupted! Using results collected so far.");
                                break;
                            }
                            eprintln!("  warning: tagger batch error: {}", e);
                            continue;
                        }
                    };
                    total_sentences_tagged += tagged_batch.len();

                    // Scan tagged output for target bigrams → accumulate votes
                    for tokens in &tagged_batch {
                        for j in 0..tokens.len().saturating_sub(1) {
                            let tw0 = tokens[j].word.to_lowercase();
                            let tw1 = tokens[j + 1].word.to_lowercase();
                            let key = (tw0, tw1);
                            if !spacy_pair_set.contains(&key) {
                                continue;
                            }
                            let patterns = match (tokens[j].pos, tokens[j + 1].pos) {
                                (POS::Adj, POS::Noun) => {
                                    vec![PatternCode::AdjNoun, PatternCode::AdjObject]
                                }
                                (POS::Verb, POS::Noun) => {
                                    vec![PatternCode::VerbNoun, PatternCode::VerbObject]
                                }
                                (POS::Noun, POS::Verb) => {
                                    vec![PatternCode::NounVerb, PatternCode::SubjVerb]
                                }
                                (POS::Prep, POS::Noun) => vec![PatternCode::PrepNoun],
                                (POS::Noun, POS::Noun) => {
                                    vec![PatternCode::NounNoun, PatternCode::PrepNoun]
                                }
                                (POS::Adv, POS::Verb) => vec![PatternCode::AdvVerb],
                                (POS::Verb, POS::Adv) => vec![PatternCode::VerbAdv],
                                (POS::Adv, POS::Adj) => vec![PatternCode::AdvAdj],
                                _ => vec![],
                            };
                            let entry = pair_votes.entry(key).or_default();
                            for pat in patterns {
                                *entry.entry(pat).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
            // Drop batch sentences + tagged output (already dropped by scope)

            total_books += batch_len;

            // Progress with ETA
            let elapsed = tag_start.elapsed().as_secs_f64();
            let rate = total_books as f64 / elapsed;
            let remaining_books = pass1.total_books.saturating_sub(total_books);
            let eta = remaining_books as f64 / rate;
            let pct = 100.0 * total_books as f64 / pass1.total_books as f64;
            eprint!(
                "\r  Pass 2: {}/{} ({:.0}%) | {} sent tagged | {:.0} books/s | ETA {}\x1b[K",
                total_books, pass1.total_books, pct,
                total_sentences_tagged, rate, fmt_duration(eta),
            );

            if remaining == Some(batch_len) {
                break;
            }
        }

        if max_books.map(|m| total_books >= m).unwrap_or(false) {
            break;
        }
    }
    eprintln!(); // newline after progress

    eprintln!(
        "  Extracted {} sentences, tagged {} | {} pairs got votes out of {}",
        total_sentences_extracted, total_sentences_tagged,
        pair_votes.len(), spacy_pair_set.len()
    );

    // Convert votes to PMI-scored counts
    let mut spacy_classified = 0usize;
    let mut spacy_skipped = 0usize;
    for (pair, votes) in &pair_votes {
        if votes.is_empty() {
            spacy_skipped += 1;
            continue;
        }
        spacy_classified += 1;
        let (w0, w1) = pair;
        let uw0 = pass1.unigram_counts.get(w0.as_str()).copied().unwrap_or(0);
        let uw1 = pass1.unigram_counts.get(w1.as_str()).copied().unwrap_or(0);
        for (pat, &vote_count) in votes {
            let freq = vote_count as u64;
            let score = pmi(freq, uw0, uw1, pass1.total_bigrams);
            let (headword, collocate) = headword_collocate(pat, w0, w1);
            let entry = counts
                .entry((headword.to_string(), *pat, collocate.to_string()))
                .or_insert(ScoreEntry { pmi: 0.0, count: 0 });
            if score > entry.pmi {
                entry.pmi = score;
                entry.count = freq;
            }
        }
    }
    // Count pairs with no examples as skipped
    spacy_skipped += total - pair_votes.len();

    eprintln!(
        "Pass 2 done: {} dict + {} tagger-lookup + {} tagged, {} skipped → {} triples",
        dict_classified, tagger_classified, spacy_classified,
        dict_skipped + spacy_skipped, counts.len()
    );
    counts
}

// ---------------------------------------------------------------------------
// Serialization: words.txt + .dat shards
// ---------------------------------------------------------------------------

/// Character code for a DictPOS used in the .dat POS field.
fn pos_char(pos: DictPOS) -> char {
    match pos {
        DictPOS::Noun => 'n',
        DictPOS::Verb => 'v',
        DictPOS::Adj => 'a',
        DictPOS::Adv => 'a', // adverbs shouldn't be headwords, fallback
        DictPOS::Prep => 'n', // preps shouldn't be headwords, fallback
    }
}

/// Ordering for POS (noun < verb < adj) for deterministic output.
fn pos_order(pos: DictPOS) -> u8 {
    match pos {
        DictPOS::Noun => 0,
        DictPOS::Verb => 1,
        DictPOS::Adj => 2,
        DictPOS::Adv => 3,
        DictPOS::Prep => 4,
    }
}

/// Write words.txt and sharded .dat files per FORMAT.md.
pub fn serialize_shards(
    output_dir: &Path,
    word_list: &[String],
    counts: &CollocationCounts,
    definitions: &HashMap<String, HashMap<DictPOS, Vec<String>>>,
    top_n: usize,
) {
    fs::create_dir_all(output_dir).expect("failed to create output directory");

    // Build word → ID mapping
    let word_to_id: HashMap<&str, usize> = word_list
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i))
        .collect();

    // Write words.txt
    {
        let words_path = output_dir.join("words.txt");
        let file = File::create(&words_path).expect("failed to create words.txt");
        let mut writer = BufWriter::new(file);
        for word in word_list {
            writeln!(writer, "{}", word).unwrap();
        }
        eprintln!("Wrote {} words to {}", word_list.len(), words_path.display());
    }

    // Group by (headword, headword_pos): → pattern → Vec<(collocate, pmi_score, count)>
    let mut by_headword_pos: HashMap<(&str, DictPOS), HashMap<PatternCode, Vec<(&str, f64, u64)>>> =
        HashMap::new();

    for ((headword, pattern, collocate), entry) in counts {
        if entry.pmi <= 0.0 {
            continue;
        }
        let hw_pos = pattern.headword_pos();
        by_headword_pos
            .entry((headword.as_str(), hw_pos))
            .or_default()
            .entry(*pattern)
            .or_default()
            .push((collocate.as_str(), entry.pmi, entry.count));
    }

    // Group headword entries by shard prefix (first 2 chars)
    let mut shards: HashMap<String, Vec<(&str, DictPOS)>> = HashMap::new();
    for &(headword, pos) in by_headword_pos.keys() {
        if headword.len() >= 2 {
            let prefix: String = headword.chars().take(2).collect();
            shards.entry(prefix).or_default().push((headword, pos));
        }
    }

    // Helper: serialize a list of entries
    let write_entries = |writer: &mut BufWriter<File>, entries: &[(&str, f64, u64)], word_to_id: &HashMap<&str, usize>| {
        for (j, (collocate, pmi_score, count)) in entries.iter().enumerate() {
            if j > 0 {
                write!(writer, ";").unwrap();
            }
            let col_id = word_to_id.get(collocate).copied().unwrap_or(0);
            let pmi_int = (pmi_score * 10.0).round() as i64;
            write!(writer, "{},{},{}", base36(col_id), pmi_int, count).unwrap();
        }
    };

    // Write .dat files
    let mut shard_count = 0;
    let mut total_lines = 0;
    for (prefix, mut shard_entries) in shards {
        // Sort by (word, pos_order) for deterministic output
        shard_entries.sort_by(|a, b| a.0.cmp(b.0).then_with(|| pos_order(a.1).cmp(&pos_order(b.1))));
        let dat_path = output_dir.join(format!("{}.dat", prefix));
        let file = File::create(&dat_path).expect("failed to create .dat file");
        let mut writer = BufWriter::new(file);

        for &(headword, hw_pos) in &shard_entries {
            let word_id = match word_to_id.get(headword) {
                Some(&id) => id,
                None => continue,
            };
            let def = definitions
                .get(headword)
                .and_then(|pos_defs| pos_defs.get(&hw_pos))
                .or_else(|| {
                    // Fallback: try any definition for this word
                    definitions.get(headword).and_then(|pos_defs| pos_defs.values().next())
                })
                .map(|defs| clean_definitions(defs))
                .unwrap_or_default();

            // Header: id|pos|definition
            write!(writer, "{}|{}|{}", base36(word_id), pos_char(hw_pos), def).unwrap();

            // Pattern groups (sorted by canonical order)
            let patterns = match by_headword_pos.get(&(headword, hw_pos)) {
                Some(p) => p,
                None => {
                    writeln!(writer).unwrap();
                    continue;
                }
            };

            let mut pattern_codes: Vec<PatternCode> = patterns.keys().copied().collect();
            pattern_codes.sort_by_key(|p| p.order());

            for pat in pattern_codes {
                let all = &patterns[&pat];
                if all.is_empty() {
                    continue;
                }

                // Top N by PMI
                let mut by_pmi = all.clone();
                by_pmi.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                by_pmi.truncate(top_n);

                // Top N by count
                let mut by_count = all.clone();
                by_count.sort_by(|a, b| b.2.cmp(&a.2));
                by_count.truncate(top_n);

                write!(writer, "\t{}:", pat.code_char()).unwrap();
                write_entries(&mut writer, &by_pmi, &word_to_id);
                write!(writer, "~").unwrap();
                write_entries(&mut writer, &by_count, &word_to_id);
            }

            writeln!(writer).unwrap();
            total_lines += 1;
        }

        shard_count += 1;
    }

    eprintln!(
        "Wrote {} shard files ({} lines) to {}",
        shard_count, total_lines, output_dir.display()
    );
}

/// Join multiple definitions with " ; ", collapsing whitespace within each.
fn clean_definitions(defs: &[String]) -> String {
    defs.iter()
        .map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join(" ; ")
}
