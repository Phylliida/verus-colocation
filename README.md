# verus-colocation
Formally verified collocation dictionary builder in Rust + Verus.

POS-tags a raw corpus via spaCy (through Rust-Python bindings), extracts collocations grouped by syntactic pattern, ranks by frequency, and outputs structured dictionary entries.

## Running

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
    --corpus data/project_gutenberg-dolma-0000.json.gz \
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
| `--corpus` | *(required)* | Path to corpus `.json.gz` file |
| `--output` | *(required)* | Output directory for `words.txt` + `.dat` shards |
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
