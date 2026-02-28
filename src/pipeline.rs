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
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::tagger::{SpacyTagger, POS};

// ---------------------------------------------------------------------------
// Pattern code (FORMAT.md single-character codes)
// ---------------------------------------------------------------------------

/// Single-character pattern codes from FORMAT.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCode {
    AdjNoun,  // 'a'
    VerbNoun, // 'v'
    NounVerb, // 'V'
    PrepNoun, // 'n'
    NounNoun, // 'N'
}

impl PatternCode {
    pub fn code_char(self) -> char {
        match self {
            PatternCode::AdjNoun => 'a',
            PatternCode::VerbNoun => 'v',
            PatternCode::NounVerb => 'V',
            PatternCode::PrepNoun => 'n',
            PatternCode::NounNoun => 'N',
        }
    }

    /// Canonical ordering for output (a, v, V, n, N).
    pub fn order(self) -> u8 {
        match self {
            PatternCode::AdjNoun => 0,
            PatternCode::VerbNoun => 1,
            PatternCode::NounVerb => 2,
            PatternCode::PrepNoun => 3,
            PatternCode::NounNoun => 4,
        }
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
    /// word → first definition.
    pub definitions: HashMap<String, String>,
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

    let mut definitions: HashMap<String, String> = HashMap::new();
    let mut pos_sets: HashMap<String, HashSet<DictPOS>> = HashMap::new();

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

        definitions.entry(word.clone()).or_insert(definition);
        let pos_list = map_wordtype(wordtype);
        if !pos_list.is_empty() {
            let set = pos_sets.entry(word).or_default();
            for pos in pos_list {
                set.insert(pos);
            }
        }
    }

    let mut words: Vec<String> = definitions.keys().cloned().collect();
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
        (DictPOS::Adj, DictPOS::Noun) => patterns.push(PatternCode::AdjNoun),
        (DictPOS::Verb, DictPOS::Noun) => patterns.push(PatternCode::VerbNoun),
        (DictPOS::Noun, DictPOS::Verb) => patterns.push(PatternCode::NounVerb),
        (DictPOS::Prep, DictPOS::Noun) => patterns.push(PatternCode::PrepNoun),
        (DictPOS::Noun, DictPOS::Noun) => {
            patterns.push(PatternCode::NounNoun);
            patterns.push(PatternCode::PrepNoun);
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
                | (DictPOS::Noun, DictPOS::Noun) => return true,
                _ => {}
            }
        }
    }
    false
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
    let lower = trimmed.to_lowercase();
    if lower.chars().all(|c| c.is_ascii_alphabetic()) {
        Some(lower)
    } else {
        None
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

/// Result of Pass 1: bigram counts, unigram counts, and example contexts.
pub struct Pass1Result {
    /// (word_a, word_b) → raw count (word_a appears before word_b).
    pub bigram_counts: HashMap<(String, String), u64>,
    /// word → total occurrence count (as part of any non-stopword bigram).
    pub unigram_counts: HashMap<String, u64>,
    /// Total number of bigrams counted (for PMI normalization).
    pub total_bigrams: u64,
    /// (word_a, word_b) → example sentence containing the pair (for POS tagging).
    /// We store up to `max_examples` sentences per pair.
    pub examples: HashMap<(String, String), Vec<String>>,
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
    /// word string → ID
    word_to_id: HashMap<String, u32>,
    /// ID → word string
    id_to_word: Vec<String>,
}

impl WordInterner {
    /// Build interner from the dictionary word set (only these words matter).
    fn from_dict(dict_words: &HashMap<String, usize>, stopwords: &HashSet<String>) -> Self {
        let mut word_to_id = HashMap::new();
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

/// Per-book accumulator using interned (u32, u32) keys.
struct BookResult {
    bigram_counts: HashMap<(u32, u32), u64>,
    unigram_counts: HashMap<u32, u64>,
    total_bigrams: u64,
    /// (u32, u32) → Vec<example context string>
    examples: HashMap<(u32, u32), Vec<String>>,
}

/// Process a single book's text, returning interned bigram counts.
fn process_book(
    text: &str,
    interner: &WordInterner,
    max_examples: usize,
) -> BookResult {
    let mut bigram_counts: HashMap<(u32, u32), u64> = HashMap::new();
    let mut unigram_counts: HashMap<u32, u64> = HashMap::new();
    let mut total_bigrams: u64 = 0;
    let mut examples: HashMap<(u32, u32), Vec<String>> = HashMap::new();

    for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
        let raw_words: Vec<&str> = sentence.split_whitespace().collect();

        // Intern tokens: normalize and look up in interner
        let token_ids: Vec<Option<u32>> = raw_words
            .iter()
            .map(|w| normalize_token(w).and_then(|n| interner.get(&n)))
            .collect();

        for i in 0..token_ids.len().saturating_sub(1) {
            if let (Some(id0), Some(id1)) = (token_ids[i], token_ids[i + 1]) {
                let key = (id0, id1);
                *bigram_counts.entry(key).or_insert(0) += 1;
                *unigram_counts.entry(id0).or_insert(0) += 1;
                *unigram_counts.entry(id1).or_insert(0) += 1;
                total_bigrams += 1;

                let exs = examples.entry(key).or_default();
                if exs.len() < max_examples {
                    let start = i.saturating_sub(3);
                    let end = (i + 5).min(raw_words.len());
                    exs.push(raw_words[start..end].join(" "));
                }
            }
        }
    }

    BookResult { bigram_counts, unigram_counts, total_bigrams, examples }
}

/// Pass 1: parallel corpus processing with interned word IDs.
///
/// 1. Read all JSON lines from gzip (sequential I/O)
/// 2. Process books in parallel with rayon (tokenize + count)
/// 3. Merge per-book results and convert back to string keys
pub fn pass1_count_bigrams(
    corpus_path: &Path,
    dict_words: &HashMap<String, usize>,
    stopwords: &HashSet<String>,
    max_books: Option<usize>,
    max_examples: usize,
) -> Pass1Result {
    // Build interner from dictionary
    let interner = WordInterner::from_dict(dict_words, stopwords);
    eprintln!("Interner: {} words indexed", interner.id_to_word.len());

    // Read all book records (I/O is sequential due to gzip)
    let file = File::open(corpus_path).expect("failed to open corpus file");
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut records: Vec<GutenbergRecord> = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("warning: skipping line due to read error: {}", e);
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(r) => records.push(r),
            Err(e) => eprintln!("warning: skipping malformed JSON: {}", e),
        }
        if let Some(max) = max_books {
            if records.len() >= max {
                eprintln!("Reached --max-books limit ({})", max);
                break;
            }
        }
    }
    eprintln!("Read {} books, processing in parallel...", records.len());

    // Process books in parallel
    let progress = AtomicUsize::new(0);
    let total_books = records.len();

    let book_results: Vec<BookResult> = records
        .par_iter()
        .map(|record| {
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            let title = record.metadata.title.as_deref().unwrap_or("(untitled)");
            if n % 50 == 0 || n == total_books {
                eprintln!("[{:>4}/{}] {}", n, total_books, title);
            }
            process_book(&record.text, &interner, max_examples)
        })
        .collect();

    // Merge results
    eprintln!("Merging results from {} books...", book_results.len());
    let mut bigram_counts: HashMap<(u32, u32), u64> = HashMap::new();
    let mut unigram_counts: HashMap<u32, u64> = HashMap::new();
    let mut total_bigrams: u64 = 0;
    let mut examples: HashMap<(u32, u32), Vec<String>> = HashMap::new();

    for br in book_results {
        total_bigrams += br.total_bigrams;
        for (key, count) in br.bigram_counts {
            *bigram_counts.entry(key).or_insert(0) += count;
        }
        for (id, count) in br.unigram_counts {
            *unigram_counts.entry(id).or_insert(0) += count;
        }
        for (key, mut exs) in br.examples {
            let dest = examples.entry(key).or_default();
            if dest.len() < max_examples {
                let take = max_examples - dest.len();
                exs.truncate(take);
                dest.extend(exs);
            }
        }
    }

    // Convert interned IDs back to strings
    let str_bigram_counts: HashMap<(String, String), u64> = bigram_counts
        .into_iter()
        .map(|((a, b), c)| ((interner.word(a).to_string(), interner.word(b).to_string()), c))
        .collect();
    let str_unigram_counts: HashMap<String, u64> = unigram_counts
        .into_iter()
        .map(|(id, c)| (interner.word(id).to_string(), c))
        .collect();
    let str_examples: HashMap<(String, String), Vec<String>> = examples
        .into_iter()
        .map(|((a, b), v)| ((interner.word(a).to_string(), interner.word(b).to_string()), v))
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
        examples: str_examples,
    }
}

// ---------------------------------------------------------------------------
// Pass 2: POS-tag candidates to classify patterns
// ---------------------------------------------------------------------------

/// Triple key: (headword, pattern, collocate) → PMI score.
pub type CollocationCounts = HashMap<(String, PatternCode, String), f64>;

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
        PatternCode::AdjNoun => (w1, w0),
        PatternCode::VerbNoun => (w1, w0),
        PatternCode::NounVerb => (w0, w1),
        PatternCode::PrepNoun => (w1, w0),
        PatternCode::NounNoun => (w0, w1),
    }
}

/// Pass 2: for each top-frequency bigram, classify its pattern.
///
/// Fast path: if both words have unambiguous dictionary POS, classify directly.
/// Slow path: POS-tag example sentences via spaCy for ambiguous pairs.
///
/// `top_per_word`: only classify the top N bigrams per word (by count).
pub fn pass2_classify(
    pass1: &Pass1Result,
    tagger: &SpacyTagger,
    pos_sets: &HashMap<String, HashSet<DictPOS>>,
    min_count: u64,
    top_per_word: usize,
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

    // Pre-classify: resolve dict-only pairs upfront, collect spaCy-needed pairs
    let mut counts: CollocationCounts = HashMap::new();
    let mut spacy_needed: Vec<(&(String, String), u64)> = Vec::new();
    let mut dict_classified = 0usize;
    let mut dict_skipped = 0usize;

    for (&pair, &raw_count) in &selected {
        let (w0, w1) = pair;
        match classify_from_dict(w0, w1, pos_sets) {
            Some(pats) if pats.is_empty() => {
                // Known POS but not a collocation pattern (e.g. ADV+ADV)
                dict_skipped += 1;
            }
            Some(pats) => {
                // Both unambiguous, classified without spaCy
                dict_classified += 1;
                let uw0 = pass1.unigram_counts.get(w0).copied().unwrap_or(0);
                let uw1 = pass1.unigram_counts.get(w1).copied().unwrap_or(0);
                let score = pmi(raw_count, uw0, uw1, pass1.total_bigrams);
                for pat in &pats {
                    let (headword, collocate) = headword_collocate(pat, w0, w1);
                    let entry = counts
                        .entry((headword.to_string(), *pat, collocate.to_string()))
                        .or_insert(0.0);
                    if score > *entry {
                        *entry = score;
                    }
                }
            }
            None => {
                // Ambiguous or missing POS — check if any combo could work
                if any_pattern_possible(w0, w1, pos_sets) {
                    spacy_needed.push((pair, raw_count));
                } else {
                    dict_skipped += 1;
                }
            }
        }
    }
    spacy_needed.sort_by(|a, b| b.1.cmp(&a.1));

    eprintln!(
        "Pass 2: {} total candidates — {} dict-classified, {} dict-skipped, {} need spaCy",
        selected.len(), dict_classified, dict_skipped, spacy_needed.len()
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

    // Now run spaCy only on the ambiguous pairs — deduplicated + batched
    let mut spacy_classified = 0usize;
    let mut spacy_skipped = 0usize;
    let total = spacy_needed.len();

    // Build lookup: which pairs are we looking for? (owned keys for easy lookup)
    let spacy_pair_set: HashSet<(String, String)> =
        spacy_needed.iter().map(|(pair, _)| (pair.0.clone(), pair.1.clone())).collect();
    let spacy_pair_counts: HashMap<(String, String), u64> =
        spacy_needed.iter().map(|(pair, count)| ((pair.0.clone(), pair.1.clone()), *count)).collect();

    // Collect all unique example sentences across all spaCy-needed pairs
    let mut unique_sentences: Vec<String> = Vec::new();
    let mut seen_sentences: HashSet<String> = HashSet::new();

    for (pair, _) in &spacy_needed {
        if let Some(examples) = pass1.examples.get(*pair) {
            for ex in examples {
                if seen_sentences.insert(ex.clone()) {
                    unique_sentences.push(ex.clone());
                }
            }
        }
    }
    drop(seen_sentences);

    eprintln!(
        "  {} unique sentences (deduplicated) for {} pairs through spaCy...",
        unique_sentences.len(), total
    );

    // Tag all unique sentences in batches via nlp.pipe()
    let spacy_batch_size = 100_000;
    let mut pair_votes: HashMap<(String, String), HashMap<PatternCode, u32>> = HashMap::new();
    let mut sentences_tagged = 0usize;

    for chunk_start in (0..unique_sentences.len()).step_by(spacy_batch_size) {
        let chunk_end = (chunk_start + spacy_batch_size).min(unique_sentences.len());
        let text_refs: Vec<&str> = unique_sentences[chunk_start..chunk_end]
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
                eprintln!("  warning: spaCy batch error: {}", e);
                continue;
            }
        };
        sentences_tagged += tagged_batch.len();

        // Scan each tagged sentence for ALL spaCy-needed bigrams it contains
        for tokens in &tagged_batch {
            for j in 0..tokens.len().saturating_sub(1) {
                let tw0 = tokens[j].word.to_lowercase();
                let tw1 = tokens[j + 1].word.to_lowercase();
                let key = (tw0, tw1);
                if !spacy_pair_set.contains(&key) {
                    continue;
                }
                let patterns = match (tokens[j].pos, tokens[j + 1].pos) {
                    (POS::Adj, POS::Noun) => vec![PatternCode::AdjNoun],
                    (POS::Verb, POS::Noun) => vec![PatternCode::VerbNoun],
                    (POS::Noun, POS::Verb) => vec![PatternCode::NounVerb],
                    (POS::Prep, POS::Noun) => vec![PatternCode::PrepNoun],
                    (POS::Noun, POS::Noun) => {
                        vec![PatternCode::NounNoun, PatternCode::PrepNoun]
                    }
                    _ => vec![],
                };
                let entry = pair_votes.entry(key).or_default();
                for pat in patterns {
                    *entry.entry(pat).or_insert(0) += 1;
                }
            }
        }

        if sentences_tagged % 2000 < spacy_batch_size || chunk_end == unique_sentences.len() {
            eprintln!("  [{}/{}] sentences tagged...", sentences_tagged, unique_sentences.len());
        }
    }

    // Convert votes to PMI-scored counts
    for (pair, votes) in &pair_votes {
        if votes.is_empty() {
            spacy_skipped += 1;
            continue;
        }
        spacy_classified += 1;
        let (w0, w1) = pair;
        let raw_count = spacy_pair_counts.get(pair).copied().unwrap_or(0);
        let uw0 = pass1.unigram_counts.get(w0.as_str()).copied().unwrap_or(0);
        let uw1 = pass1.unigram_counts.get(w1.as_str()).copied().unwrap_or(0);
        let score = pmi(raw_count, uw0, uw1, pass1.total_bigrams);
        for (pat, _) in votes {
            let (headword, collocate) = headword_collocate(pat, w0, w1);
            let entry = counts
                .entry((headword.to_string(), *pat, collocate.to_string()))
                .or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }
    }
    // Count pairs with no examples as skipped
    spacy_skipped += total - pair_votes.len();

    eprintln!(
        "Pass 2 done: {} dict + {} spaCy classified, {} skipped → {} triples",
        dict_classified, spacy_classified, dict_skipped + spacy_skipped, counts.len()
    );
    counts
}

// ---------------------------------------------------------------------------
// Serialization: words.txt + .dat shards
// ---------------------------------------------------------------------------

/// Write words.txt and sharded .dat files per FORMAT.md.
pub fn serialize_shards(
    output_dir: &Path,
    word_list: &[String],
    counts: &CollocationCounts,
    definitions: &HashMap<String, String>,
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

    // Group by headword: headword → pattern → Vec<(collocate, pmi_score)>
    let mut by_headword: HashMap<&str, HashMap<PatternCode, Vec<(&str, f64)>>> = HashMap::new();

    for ((headword, pattern, collocate), &score) in counts {
        if score <= 0.0 {
            continue;
        }
        by_headword
            .entry(headword.as_str())
            .or_default()
            .entry(*pattern)
            .or_default()
            .push((collocate.as_str(), score));
    }

    // Sort collocates within each pattern by PMI descending, take top_n
    for patterns in by_headword.values_mut() {
        for entries in patterns.values_mut() {
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(top_n);
        }
    }

    // Group headwords by shard prefix (first 2 chars)
    let mut shards: HashMap<String, Vec<&str>> = HashMap::new();
    for &headword in by_headword.keys() {
        if headword.len() >= 2 {
            let prefix: String = headword.chars().take(2).collect();
            shards.entry(prefix).or_default().push(headword);
        }
    }

    // Write .dat files
    let mut shard_count = 0;
    for (prefix, mut headwords) in shards {
        headwords.sort();
        let dat_path = output_dir.join(format!("{}.dat", prefix));
        let file = File::create(&dat_path).expect("failed to create .dat file");
        let mut writer = BufWriter::new(file);

        for &headword in &headwords {
            let word_id = match word_to_id.get(headword) {
                Some(&id) => id,
                None => continue,
            };
            let def = definitions
                .get(headword)
                .map(|d| truncate_definition(d, 80))
                .unwrap_or_default();

            // Header: id|ipa|definition
            // IPA is left empty (we don't have pronunciation data)
            write!(writer, "{}||{}", base36(word_id), def).unwrap();

            // Pattern groups (sorted by canonical order)
            let patterns = match by_headword.get(headword) {
                Some(p) => p,
                None => {
                    writeln!(writer).unwrap();
                    continue;
                }
            };

            let mut pattern_codes: Vec<PatternCode> = patterns.keys().copied().collect();
            pattern_codes.sort_by_key(|p| p.order());

            for pat in pattern_codes {
                let entries = &patterns[&pat];
                if entries.is_empty() {
                    continue;
                }
                write!(writer, "\t{}:", pat.code_char()).unwrap();
                for (j, (collocate, score)) in entries.iter().enumerate() {
                    if j > 0 {
                        write!(writer, ";").unwrap();
                    }
                    let col_id = word_to_id.get(collocate).copied().unwrap_or(0);
                    // FORMAT.md: score × 10, stored as integer
                    let score_int = (score * 10.0).round() as i64;
                    write!(writer, "{},{}", base36(col_id), score_int).unwrap();
                }
            }

            writeln!(writer).unwrap();
        }

        shard_count += 1;
    }

    eprintln!("Wrote {} shard files to {}", shard_count, output_dir.display());
}

/// Truncate a definition to at most `max_len` characters, breaking at word boundary.
fn truncate_definition(def: &str, max_len: usize) -> String {
    // Collapse whitespace first
    let clean: String = def.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.len() <= max_len {
        return clean;
    }
    let truncated = &clean[..max_len];
    match truncated.rfind(' ') {
        Some(pos) => format!("{}...", &truncated[..pos]),
        None => format!("{}...", truncated),
    }
}
