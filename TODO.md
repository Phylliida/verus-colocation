# verus-colocation: Burndown

## Phase 1: Data Model (verified core types)
- [x] Define POS tag enum (Adj, Noun, Verb, Adv, Prep, Det, Other)
- [x] Define Pattern enum (AdjNoun, VerbNoun, NounVerb, PrepNoun, NounNoun)
- [x] Define `Collocation` struct: pattern + position index
- [ ] Define `DictionaryEntry` struct: headword + grouped collocations by pattern

## Phase 2: Collocation Extraction (verified) — 6 verified, 0 errors
- [x] Linear-pass bigram pattern matcher over `Vec<POS>`
- [x] Extract adj+noun, verb+noun, noun+verb collocations
- [x] Spec predicates: `is_adj_noun_at`, `is_verb_noun_at`, `is_noun_verb_at`
- [x] Proof: completeness (every collocation position captured)
- [x] Proof: soundness (every result matches a real collocation)
- [x] Proof: bounds safe (all positions valid)
- [x] Proof: ascending positions (no duplicates)

## Phase 3: Frequency & Ranking (verified) — 12 verified total, 0 errors
- [x] `RankedEntry` struct (pattern + pos + u64 count)
- [x] Spec: `is_pairwise_descending`, `all_above_threshold`
- [x] Exec: `filter_by_threshold` with verified threshold property
- [x] Exec: `sort_descending` (selection sort, verified pairwise descending)
- [x] Proof: `lemma_pairwise_implies_sorted` (pairwise ⇒ global order)
- [x] Proof: sort preserves partition invariant through swap
- [x] No trusted escapes (assume, admit, unsafe, external)

## Phase 4: spaCy Tagger Integration (exec, unverified)
- [x] pyo3 0.23 dependency (feature-gated: `--features tagger`)
- [x] `tagger` module: `SpacyTagger` loads model, tags text → `Vec<TaggedToken>`
- [x] `tag_pos_only()` bridge: returns `Vec<POS>` for verified core
- [x] POS mapping: spaCy Universal tags → our POS enum (ADJ, NOUN, PROPN→Noun, VERB, AUX→Verb, etc.)
- [x] Integration tests passing (`busy bees buzz loudly`, `The quick brown fox`)

## Phase 5: Output (exec, unverified)
- [ ] HTML output matching mockup.html format
- [ ] JSON output for programmatic use
- [x] CLI entry point: `src/bin/generate.rs` — two-pass pipeline (bigram count → POS classify)
- [x] Pipeline module: `src/pipeline.rs` — dictionary parsing, bigram counting, POS classification, shard serialization
- [x] FORMAT.md-compliant output: words.txt + sharded .dat files

## Infrastructure
- [x] Cargo.toml + src stubs
- [x] CI workflow (.github/workflows/check.yml) — min-verified 12
- [x] scripts/check.sh + check-single-file.sh
- [x] README with verification command
- [x] No trusted escapes (assume, admit, unsafe, external)

## Environment notes
- `PYO3_PYTHON=/Users/yams/.venv/bin/python3` for build
- `PYTHONPATH=/Users/yams/.venv/lib/python3.10/site-packages` for runtime
- spaCy model: `en_core_web_sm` (install via `python -m spacy download en_core_web_sm`)
