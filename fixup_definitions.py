#!/usr/bin/env python3
"""
Fixup script for output-data/*.dat files.

Bug: each POS entry for a word shows ALL definitions (or the noun fallback)
     instead of only the definitions for that specific POS.

Fix: re-read dictionary.csv, rebuild per-POS definitions, and rewrite
     the definition field in each .dat line.
"""

import csv
import os
import sys
from collections import defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output-data")
DICT_PATH = os.path.join(os.path.dirname(__file__), "dictionary.csv")
WORDS_PATH = os.path.join(OUTPUT_DIR, "words.txt")


def map_wordtype(wt: str) -> list[str]:
    """Mirror the Rust map_wordtype logic. Returns list of pos chars (n/v/a)."""
    wt = wt.strip().rstrip(".")
    mapping = {
        "n": ["n"], "n. pl": ["n"], "pl": ["n"],
        "a": ["a"], "superl": ["a"],
        "v": ["v"], "v. t": ["v"], "v. i": ["v"], "v. t. & i": ["v"],
        "adv": ["a"],      # adv maps to 'a' char (DictPOS::Adv → pos_char 'a')
        "prep": ["n"], "conj": ["n"],  # prep/conj map to 'n' char
        "a. & n": ["a", "n"],
        "n. & v": ["n", "v"],
        "a. & adv": ["a"],  # both map to 'a' char
        "imp. & p. p": ["v", "a"], "p. p": ["v", "a"], "imp": ["v", "a"],
        "p. pr. & vb. n": ["v", "n"],
        "p. p. & a": ["a", "v"],
    }
    return mapping.get(wt, [])


def clean_definition(d: str) -> str:
    """Collapse whitespace within a definition."""
    return " ".join(d.split())


def parse_dictionary(path: str) -> dict[str, dict[str, list[str]]]:
    """
    Parse dictionary.csv → {word: {pos_char: [definitions]}}.
    Only stores definitions under the POS they belong to.
    Empty wordtype entries are stored under a special key '' for later use.
    """
    defs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            word = row[0].strip().lower()
            if not word or not word.isascii() or not word.isalpha():
                continue
            wordtype = row[1].strip()
            definition = row[2].strip()
            if not definition:
                continue
            definition = clean_definition(definition)

            pos_chars = map_wordtype(wordtype)
            if pos_chars:
                for pc in set(pos_chars):  # deduplicate chars
                    if definition not in defs[word][pc]:
                        defs[word][pc].append(definition)
            else:
                # No POS mapped — store under '' (untagged)
                if definition not in defs[word][""]:
                    defs[word][""].append(definition)

    return dict(defs)


def resolve_definition(word_defs: dict[str, list[str]], pos_char: str) -> str:
    """
    Get the definition string for a word+pos_char.
    Priority:
      1. Definitions explicitly tagged for this POS
      2. Untagged definitions (empty wordtype in CSV)
      3. Empty string (no definition available)
    Does NOT fall back to other POS definitions.
    """
    tagged = word_defs.get(pos_char, [])
    untagged = word_defs.get("", [])

    if tagged and untagged:
        # Combine: tagged first, then untagged
        combined = tagged + [d for d in untagged if d not in tagged]
        return " ; ".join(combined)
    elif tagged:
        return " ; ".join(tagged)
    elif untagged:
        return " ; ".join(untagged)
    else:
        return ""


def decode_base36(s: str) -> int:
    return int(s, 36)


def main():
    if not os.path.exists(DICT_PATH):
        print(f"Error: dictionary not found at {DICT_PATH}", file=sys.stderr)
        sys.exit(1)

    # 1. Load word list (line number = word ID)
    with open(WORDS_PATH) as f:
        id_to_word = {i: line.strip() for i, line in enumerate(f)}
    print(f"Loaded {len(id_to_word)} words from words.txt", file=sys.stderr)

    # 2. Parse dictionary with per-POS definitions
    defs = parse_dictionary(DICT_PATH)
    print(f"Parsed definitions for {len(defs)} words from dictionary.csv", file=sys.stderr)

    # 3. Process each .dat file
    dat_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".dat"))
    total_fixed = 0
    total_lines = 0

    for dat_name in dat_files:
        dat_path = os.path.join(OUTPUT_DIR, dat_name)
        with open(dat_path, "rb") as f:
            raw = f.read()
        lines = raw.split(b"\n")
        new_lines = []

        for line in lines:
            if not line:
                new_lines.append(line)
                continue

            # Parse: word_id|pos_char|definition\tcollocation_data...
            # The first | separates word_id, second | separates pos_char,
            # everything after the second | up to the first \t is the definition.
            text = line.decode("utf-8", errors="replace")

            first_pipe = text.index("|")
            second_pipe = text.index("|", first_pipe + 1)
            word_id_str = text[:first_pipe]
            pos_char = text[first_pipe + 1 : second_pipe]

            tab_pos = text.find("\t", second_pipe)
            if tab_pos == -1:
                old_def = text[second_pipe + 1 :]
                collocation_suffix = ""
            else:
                old_def = text[second_pipe + 1 : tab_pos]
                collocation_suffix = text[tab_pos:]  # includes the leading \t

            # Look up the word
            word_id = decode_base36(word_id_str)
            word = id_to_word.get(word_id)

            if word and word in defs:
                new_def = resolve_definition(defs[word], pos_char)
            else:
                new_def = old_def  # keep as-is if we can't resolve

            if new_def != old_def:
                total_fixed += 1

            new_line = f"{word_id_str}|{pos_char}|{new_def}{collocation_suffix}"
            new_lines.append(new_line.encode("utf-8"))
            total_lines += 1

        with open(dat_path, "wb") as f:
            f.write(b"\n".join(new_lines))

    print(
        f"Processed {len(dat_files)} .dat files, {total_lines} lines, "
        f"{total_fixed} definitions fixed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
