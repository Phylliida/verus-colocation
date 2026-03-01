# Colocation Data Format

## Source Data (`dictionary.csv`)

Standard CSV with three columns:

| Column | Description |
|--------|-------------|
| `word` | The headword (lowercase) |
| `wordtype` | Part of speech abbreviation (e.g. `n.`, `v. t.`, `a.`, `prep.`). May be empty. |
| `definition` | Full definition text. May span multiple lines and contain quoted examples. |

A word can have multiple rows (one per sense/part of speech). Definitions may contain multi-line text with leading whitespace.

```
word,wordtype,definition
a,prep.,In; on; at; by.
bee,n.,"An insect of the order Hymenoptera..."
```

This file is the input used to build the word list and populate definitions in the shard files.

## Corpus Data (`data/project_gutenberg-dolma-*.json.gz`)

Gzipped JSON Lines (one JSON object per line). This is the raw text corpus used to extract collocations. The data comes from the [Dolma](https://github.com/allenai/dolma) dataset's Project Gutenberg subset.

**File stats** (for `0000` shard): 544MB compressed, 4,025 records, median text ~289KB.

Each line is a JSON object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Gutenberg ebook ID (e.g. `"1"`, `"100"`) |
| `text` | string | Full text of the book (can be very large, up to ~5MB) |
| `source` | string | Always `"project gutenberg"` |
| `added` | string | ISO 8601 timestamp of when the record was added |
| `metadata` | object | See below |

**`metadata` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Language code (all `"en"` in this shard) |
| `license` | string | Always `"Public Domain"` |
| `provenance` | string | Source filename and line number |
| `title` | string | Book title |
| `url` | string | Gutenberg source URL |

```json
{
  "id": "1",
  "text": "December, 1971  [Etext #1]\n\nThe Project Gutenberg...",
  "source": "project gutenberg",
  "added": "2024-05-14T12:56:48.592044",
  "metadata": {
    "language": "en",
    "license": "Public Domain",
    "provenance": "project_gutenberg-dolma-0000.json.gz:1",
    "title": "The Declaration of Independence of the United States of America",
    "url": "https://www.gutenberg.org/ebooks/1.txt.utf-8"
  }
}
```

## Overview

Collocation data is split into two file types: a single word list and many shard files. The word list maps words to integer IDs. Shard files store collocation data using those IDs, grouped by first two letters of the headword.

## Word List (`words.txt`)

One word per line, sorted alphabetically. The line number (0-indexed) is the word's ID.

```
a
abandon
...
bee
...
```

IDs are encoded as **base36** in shard files (e.g. `8858` → `6u2`).

## Shard Files (`{prefix}.dat`)

Each shard is named by the first two letters of the words it contains (e.g. `be.dat` for bee, beat, bear). One headword per line.

### Line format

```
{id}|{pos}|{definition}\t{pattern}:{entries}\t{pattern}:{entries}...
```

- **Header** (pipe-delimited): word ID (base36), POS tag (`n`/`v`/`a`), short definition
- **Pattern groups** (tab-separated): each is a pattern type character followed by `:` and semicolon-separated entries
- **Entry**: `{word_id},{score}` where word_id is base36 and score is an integer (actual score × 10)

A word with multiple parts of speech gets **multiple lines** (one per POS), each with POS-appropriate patterns.

### POS tags

| Tag | Part of Speech |
|-----|---------------|
| `n` | Noun |
| `v` | Verb |
| `a` | Adjective |

### Pattern types (11 total)

#### Noun headword patterns

| Code | Bigram | Headword | Collocate | Example |
|------|--------|----------|-----------|---------|
| `a` | ADJ NOUN | noun(w1) | adj(w0) | _cold_ **water** |
| `v` | VERB NOUN | noun(w1) | verb(w0) | _drink_ **water** |
| `V` | NOUN VERB | noun(w0) | verb(w1) | **water** _flows_ |
| `n` | PREP NOUN | noun(w1) | prep(w0) | _in_ **water** |
| `N` | NOUN NOUN | noun(w0) | noun(w1) | **water** _pipe_ |

#### Verb headword patterns

| Code | Bigram | Headword | Collocate | Example |
|------|--------|----------|-----------|---------|
| `o` | VERB NOUN | verb(w0) | noun(w1) | **drink** _water_ |
| `s` | NOUN VERB | verb(w1) | noun(w0) | _dogs_ **run** |
| `d` | ADV VERB | verb(w1) | adv(w0) | _quickly_ **run** |
| `D` | VERB ADV | verb(w0) | adv(w1) | **run** _quickly_ |

#### Adjective headword patterns

| Code | Bigram | Headword | Collocate | Example |
|------|--------|----------|-----------|---------|
| `j` | ADJ NOUN | adj(w0) | noun(w1) | **cold** _water_ |
| `e` | ADV ADJ | adj(w1) | adv(w0) | _very_ **cold** |

### Symmetry

One bigram type produces entries for multiple headwords:
- ADJ+NOUN → noun gets `a`, adj gets `j`
- VERB+NOUN → noun gets `v`, verb gets `o`
- NOUN+VERB → noun gets `V`, verb gets `s`
- ADV+VERB → verb gets `d`
- VERB+ADV → verb gets `D`
- ADV+ADJ → adj gets `e`
- PREP+NOUN → noun gets `n` (no other side)
- NOUN+NOUN → first noun gets `N`

### Example

Raw lines for "water" (noun and verb entries):
```
285t|n|The fluid which descends from the clouds in rain	a:...\tv:...\tV:...\tn:...\tN:...
285t|v|To wet or supply with water; to moisten	o:...\ts:...
```

Raw line for "cold" (adjective entry):
```
1a2|a|Deprived of heat; having a low temperature	j:...\te:...
```

## Sharding

Files are sharded by the first two characters of the headword. The client fetches only the shard it needs (e.g. searching "water" fetches `wa.dat`). Typical shard size with ~250 words is a few KB, compresses well with gzip/brotli.
