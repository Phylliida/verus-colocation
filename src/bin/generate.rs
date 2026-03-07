//! CLI binary for generating collocation data files.
//!
//! Two-pass pipeline:
//!   Pass 1: fast bigram counting (no POS tagger)
//!   Pass 2: POS-tag top candidates to classify patterns
//!
//! Usage:
//!   # Single file:
//!   cargo run --features tagger --bin generate -- \
//!       --dictionary dictionary.csv \
//!       --corpus data/project_gutenberg-dolma-0000.json.gz \
//!       --output output-data/ \
//!       --max-books 100 \
//!       --top-n 10 \
//!       --min-count 3
//!
//!   # Entire directory of .json.gz files:
//!   cargo run --features postagger-backend --bin generate -- \
//!       --backend rust \
//!       --dictionary dictionary.csv \
//!       --corpus data/ \
//!       --output output-data/

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use verus_colocation::pipeline;
use verus_colocation::pos::Tagger;

/// Pick the default backend based on which features are compiled in.
fn default_backend() -> &'static str {
    #[cfg(feature = "tagger")]
    { return "spacy"; }

    #[cfg(feature = "postagger-backend")]
    { return "rust"; }

    #[allow(unreachable_code)]
    {
        eprintln!("error: no tagger backend available; enable `tagger` or `postagger-backend`");
        std::process::exit(1);
    }
}

struct Args {
    dictionary: PathBuf,
    corpus_files: Vec<PathBuf>,
    output: PathBuf,
    max_books: Option<usize>,
    top_n: usize,
    min_count: u64,
    max_examples: usize,
    backend: String,
}

/// Resolve `--corpus` path: if it's a directory, collect all `*.json.gz` files
/// sorted by name; otherwise treat it as a single file.
fn resolve_corpus(path: PathBuf) -> Vec<PathBuf> {
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
            .unwrap_or_else(|e| panic!("failed to read directory {}: {}", path.display(), e))
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gz")
                    && p.to_str().map_or(false, |s| s.ends_with(".json.gz"))
                {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();
        files.sort();
        if files.is_empty() {
            eprintln!("warning: no .json.gz files found in {}", path.display());
        } else {
            eprintln!("Found {} .json.gz files in {}", files.len(), path.display());
        }
        files
    } else {
        vec![path]
    }
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
    let mut backend = default_backend().to_string();

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
            "--backend" => {
                backend = args.next().unwrap_or_else(|| {
                    eprintln!("--backend requires a value (spacy or rust)");
                    std::process::exit(1);
                });
            }
            "--help" | "-h" => {
                eprintln!("Usage: generate --dictionary <path> --corpus <path|dir> --output <dir>");
                eprintln!("  --dictionary    Path to dictionary.csv");
                eprintln!("  --corpus        Path to a .json.gz file or directory of .json.gz files");
                eprintln!("  --output        Output directory for words.txt + .dat shards");
                eprintln!("  --max-books N   Process at most N books (default: all)");
                eprintln!("  --top-n N       Keep top N collocates per pattern (default: 10)");
                eprintln!("  --min-count N   Minimum bigram count threshold (default: 3)");
                eprintln!("  --max-examples N  Example sentences to store per bigram (default: 5)");
                eprintln!("  --backend       POS tagger backend: spacy (default) or rust");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    let corpus_path = corpus.expect("--corpus is required");
    Args {
        dictionary: dictionary.expect("--dictionary is required"),
        corpus_files: resolve_corpus(corpus_path),
        output: output.expect("--output is required"),
        max_books,
        top_n,
        min_count,
        max_examples,
        backend,
    }
}

/// Instantiate the requested tagger backend.
fn make_tagger(backend: &str) -> Box<dyn Tagger> {
    match backend {
        #[cfg(feature = "tagger")]
        "spacy" => {
            eprintln!("Loading spaCy model...");
            let t = verus_colocation::tagger::SpacyTagger::new("en_core_web_sm")
                .expect("failed to load en_core_web_sm");
            Box::new(t)
        }
        #[cfg(not(feature = "tagger"))]
        "spacy" => {
            eprintln!("error: spacy backend requires the `tagger` feature");
            eprintln!("  cargo run --features tagger --bin generate -- ...");
            std::process::exit(1);
        }
        #[cfg(feature = "postagger-backend")]
        "rust" => {
            eprintln!("Loading Rust postagger model...");
            let t = verus_colocation::postagger_backend::RustTagger::new(
                "tagger-data/weights.json",
                "tagger-data/classes.txt",
                "tagger-data/tags.json",
            );
            Box::new(t)
        }
        #[cfg(not(feature = "postagger-backend"))]
        "rust" => {
            eprintln!("error: rust backend requires the `postagger-backend` feature");
            eprintln!("  cargo run --features postagger-backend --bin generate -- ...");
            std::process::exit(1);
        }
        other => {
            eprintln!("Unknown backend: {} (expected 'spacy' or 'rust')", other);
            std::process::exit(1);
        }
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

    // 2. Pass 1: count raw bigrams (fast, no tagger)
    eprintln!("\n=== Pass 1: counting bigrams ===");
    let pass1 = pipeline::pass1_count_bigrams(
        &args.corpus_files,
        &dict_words,
        &stopwords,
        args.max_books,
    );

    // 3. Pass 2: classify patterns (dict POS fast path + tagger fallback)
    eprintln!("\n=== Pass 2: classifying patterns ===");
    let tagger = make_tagger(&args.backend);
    // top_per_word: classify top bigrams per word (enough for 11 patterns × top_n entries)
    let top_per_word = args.top_n * 20;
    let counts = pipeline::pass2_classify(
        &pass1,
        &args.corpus_files,
        &dict_words,
        &stopwords,
        tagger.as_ref(),
        &dict.pos_sets,
        args.min_count,
        top_per_word,
        args.max_books,
        args.max_examples,
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
