//! CLI binary for generating collocation data files.
//!
//! Two-pass pipeline:
//!   Pass 1: fast bigram counting (no spaCy)
//!   Pass 2: POS-tag top candidates to classify patterns
//!
//! Usage:
//!   cargo run --features tagger --bin generate -- \
//!       --dictionary dictionary.csv \
//!       --corpus data/project_gutenberg-dolma-0000.json.gz \
//!       --output output-data/ \
//!       --max-books 100 \
//!       --top-n 10 \
//!       --min-count 3

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use verus_colocation::pipeline;
use verus_colocation::tagger::SpacyTagger;

struct Args {
    dictionary: PathBuf,
    corpus: PathBuf,
    output: PathBuf,
    max_books: Option<usize>,
    top_n: usize,
    min_count: u64,
    max_examples: usize,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut dictionary = None;
    let mut corpus = None;
    let mut output = None;
    let mut max_books = None;
    let mut top_n = 10usize;
    let mut min_count = 3u64;
    let mut max_examples = 5usize;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dictionary" => dictionary = args.next().map(PathBuf::from),
            "--corpus" => corpus = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--max-books" => {
                max_books = args.next().and_then(|s| s.parse().ok());
            }
            "--top-n" => {
                top_n = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(top_n);
            }
            "--min-count" => {
                min_count = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(min_count);
            }
            "--max-examples" => {
                max_examples = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(max_examples);
            }
            "--help" | "-h" => {
                eprintln!("Usage: generate --dictionary <path> --corpus <path> --output <dir>");
                eprintln!("  --dictionary    Path to dictionary.csv");
                eprintln!("  --corpus        Path to corpus .json.gz file");
                eprintln!("  --output        Output directory for words.txt + .dat shards");
                eprintln!("  --max-books N   Process at most N books (default: all)");
                eprintln!("  --top-n N       Keep top N collocates per pattern (default: 10)");
                eprintln!("  --min-count N   Minimum bigram count threshold (default: 3)");
                eprintln!("  --max-examples N  Example sentences to store per bigram (default: 5)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    Args {
        dictionary: dictionary.expect("--dictionary is required"),
        corpus: corpus.expect("--corpus is required"),
        output: output.expect("--output is required"),
        max_books,
        top_n,
        min_count,
        max_examples,
    }
}

fn main() {
    let args = parse_args();

    // 1. Parse dictionary
    eprintln!("Parsing dictionary: {}", args.dictionary.display());
    let dict = pipeline::parse_dictionary(&args.dictionary);
    eprintln!("Dictionary: {} unique words", dict.words.len());

    // Build word → index mapping for pass 1
    let dict_words: HashMap<String, usize> = dict
        .words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();

    // Load stopwords
    let stopwords_path = args.dictionary.parent().unwrap_or(Path::new(".")).join("stopwords.txt");
    let stopwords = pipeline::load_stopwords(&stopwords_path);
    eprintln!("Stopwords: {} loaded from {}", stopwords.len(), stopwords_path.display());

    // 2. Pass 1: count raw bigrams (fast, no spaCy)
    eprintln!("\n=== Pass 1: counting bigrams ===");
    let pass1 = pipeline::pass1_count_bigrams(
        &args.corpus,
        &dict_words,
        &stopwords,
        args.max_books,
        args.max_examples,
    );

    // 3. Pass 2: classify patterns (dict POS fast path + spaCy fallback)
    eprintln!("\n=== Pass 2: classifying patterns ===");
    eprintln!("Loading spaCy model...");
    let tagger =
        SpacyTagger::new("en_core_web_sm").expect("failed to load en_core_web_sm");
    // top_per_word: classify top 100 bigrams per word (enough for 5 patterns × 10 entries)
    let top_per_word = args.top_n * 10;
    let counts = pipeline::pass2_classify(
        &pass1,
        &tagger,
        &dict.pos_sets,
        args.min_count,
        top_per_word,
    );

    // 4. Serialize output
    eprintln!("\n=== Serializing output ===");
    pipeline::serialize_shards(
        &args.output,
        &dict.words,
        &counts,
        &dict.definitions,
        args.top_n,
    );

    eprintln!("\nDone!");
}
