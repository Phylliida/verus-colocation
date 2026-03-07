# verus-colocation
Formally verified collocation dictionary builder in Rust + Verus.

POS-tags a raw corpus, extracts collocations grouped by syntactic pattern, ranks by frequency, and outputs structured dictionary entries. Two tagger backends are available: spaCy (Python, via pyo3 bindings) and a pure-Rust averaged perceptron tagger (~8x faster, no Python dependency).

## Running

### Option A: Pure-Rust backend (recommended)

No Python needed. Uses the `postagger` crate (NLTK-style averaged perceptron). About 8x faster than spaCy (~8,000 sentences/s vs ~1,000 sentences/s).

```bash
cargo run --features postagger-backend --release --bin generate -- \
    --dictionary dictionary.csv \
    --corpus data/ \
    --output output-data/ \
    --max-books 100 \
    --top-n 10 \
    --min-count 3
```

### Option B: spaCy backend

Requires a Python venv with spaCy and `en_core_web_sm` installed:

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install spacy
python -m spacy download en_core_web_sm
```

Build and run (release mode):

```bash
PYO3_PYTHON=/Users/yams/.venv/bin/python3 \
PYTHONPATH=/Users/yams/.venv/lib/python3.10/site-packages \
cargo run --features tagger --release --bin generate -- \
    --dictionary dictionary.csv \
    --corpus data/ \
    --output output-data/ \
    --max-books 100 \
    --top-n 10 \
    --min-count 3
```

Note: `PYO3_PYTHON` tells PyO3 which Python to link against at compile time. `PYTHONPATH` ensures the venv's site-packages are visible at runtime.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dictionary` | *(required)* | Path to `dictionary.csv` |
| `--corpus` | *(required)* | Path to a `.json.gz` file or directory of `.json.gz` files |
| `--output` | *(required)* | Output directory for `words.txt` + `.dat` shards |
| `--backend` | auto | POS tagger backend: `spacy` or `rust` (auto-detected from features) |
| `--max-books N` | all | Process at most N books |
| `--top-n N` | 10 | Keep top N collocates per pattern |
| `--min-count N` | 3 | Minimum bigram count threshold |
| `--max-examples N` | 5 | Example sentences stored per bigram (for POS tagging) |

## Verification

Requires [Verus](https://github.com/verus-lang/verus) built in `../verus/`.

```bash
./scripts/check.sh --require-verus --forbid-trusted-escapes --min-verified 12
```

## License

MIT
