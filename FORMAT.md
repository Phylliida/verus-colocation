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
{id}|{ipa}|{definition}\t{pattern}:{entries}\t{pattern}:{entries}...
```

- **Header** (pipe-delimited): word ID (base36), IPA pronunciation, short definition
- **Pattern groups** (tab-separated): each is a pattern type character followed by `:` and semicolon-separated entries
- **Entry**: `{word_id},{score}` where word_id is base36 and score is an integer (actual score × 10)

### Pattern types

| Code | Meaning | Display |
|------|---------|---------|
| `a` | adjective + word | `busy bee` |
| `v` | verb + word | `keep bees` |
| `V` | word + verb | `bees buzz` |
| `n` | noun/prep + word | `swarm of bees` |
| `N` | word + noun | `beekeeper` |

### Example

Raw line:
```
6u2|/biː/|A flying insect known for pollination.\ta:9t6,142;yca,138\tv:1491,121\tV:9v9,135\tn:1zyr,127\tN:1xy4,104
```

Decoded:
```
bee  /biː/  A flying insect known for pollination.
  adj+:   busy 14.2, honey 13.8
  verb+:  keep 12.1
  +verb:  buzz 13.5
  noun+:  swarm 12.7
  +noun:  sting 10.4
```

## Sharding

Files are sharded by the first two characters of the headword. The client fetches only the shard it needs (e.g. searching "water" fetches `wa.dat`). Typical shard size with ~250 words is a few KB, compresses well with gzip/brotli.
