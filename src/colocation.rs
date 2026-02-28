use vstd::prelude::*;

verus! {

/// Part-of-speech tags (simplified Universal POS tagset).
#[derive(PartialEq, Eq, Structural, Copy, Clone)]
pub enum POS {
    Adj,
    Noun,
    Verb,
    Adv,
    Prep,
    Det,
    Other,
}

/// Syntactic pattern that a collocation matches.
#[derive(PartialEq, Eq, Structural, Copy, Clone)]
pub enum Pattern {
    AdjNoun,
    VerbNoun,
    NounVerb,
}

/// A collocation found at a specific position in the stream.
/// Stores the pattern and the starting index (the pair is at `pos`, `pos+1`).
pub struct Collocation {
    pub pattern: Pattern,
    pub pos: usize,
}

// ---------------------------------------------------------------------------
// Pattern matching predicates
// ---------------------------------------------------------------------------

/// True when the bigram at position `i` matches `AdjNoun`.
pub open spec fn is_adj_noun_at(tags: Seq<POS>, i: int) -> bool {
    0 <= i && i + 1 < tags.len()
    && tags[i] == POS::Adj
    && tags[i + 1] == POS::Noun
}

/// True when the bigram at position `i` matches `VerbNoun`.
pub open spec fn is_verb_noun_at(tags: Seq<POS>, i: int) -> bool {
    0 <= i && i + 1 < tags.len()
    && tags[i] == POS::Verb
    && tags[i + 1] == POS::Noun
}

/// True when the bigram at position `i` matches `NounVerb`.
pub open spec fn is_noun_verb_at(tags: Seq<POS>, i: int) -> bool {
    0 <= i && i + 1 < tags.len()
    && tags[i] == POS::Noun
    && tags[i + 1] == POS::Verb
}

/// True when the bigram at position `i` matches any supported pattern.
pub open spec fn is_collocation_at(tags: Seq<POS>, i: int) -> bool {
    is_adj_noun_at(tags, i) || is_verb_noun_at(tags, i) || is_noun_verb_at(tags, i)
}

/// Determine the pattern of the bigram at position `i`.
pub open spec fn pattern_at(tags: Seq<POS>, i: int) -> Pattern
    recommends
        is_collocation_at(tags, i),
{
    if is_adj_noun_at(tags, i) {
        Pattern::AdjNoun
    } else if is_verb_noun_at(tags, i) {
        Pattern::VerbNoun
    } else {
        Pattern::NounVerb
    }
}

// ---------------------------------------------------------------------------
// Postcondition helpers
// ---------------------------------------------------------------------------

/// True when every entry in `result` is a valid collocation in `tags`.
pub open spec fn all_entries_valid(result: Seq<Collocation>, tags: Seq<POS>) -> bool {
    forall|k: int| #![trigger result[k]] 0 <= k < result.len() ==>
        (result[k].pos as int) + 1 < tags.len()
        && result[k].pattern == pattern_at(tags, result[k].pos as int)
        && is_collocation_at(tags, result[k].pos as int)
}

/// True when positions in `result` are strictly ascending.
pub open spec fn positions_ascending(result: Seq<Collocation>) -> bool {
    forall|k: int, l: int| #![trigger result[k], result[l]]
        0 <= k < l < result.len() ==> result[k].pos < result[l].pos
}

/// True when all positions in `result` are less than `bound`.
pub open spec fn all_positions_below(result: Seq<Collocation>, bound: int) -> bool {
    forall|k: int| #![trigger result[k]] 0 <= k < result.len() ==>
        (result[k].pos as int) < bound
}

/// True when every collocation in `tags` at positions `[0, up_to)` is
/// represented in `result`.
pub open spec fn complete_up_to(
    result: Seq<Collocation>,
    tags: Seq<POS>,
    up_to: int,
) -> bool {
    forall|j: int| 0 <= j < up_to && j + 1 < tags.len()
        && is_collocation_at(tags, j) ==>
        exists|k: int| #![trigger result[k]] 0 <= k < result.len()
            && result[k].pos == j as usize
}

// ---------------------------------------------------------------------------
// Extraction: executable linear pass
// ---------------------------------------------------------------------------

/// Extract all bigram collocations from `tags` in a single left-to-right
/// pass.
///
/// Verified properties:
///   1. Bounds safety — every index access is within bounds.
///   2. Soundness — every returned position actually matches a collocation.
///   3. Ascending — the returned positions are strictly increasing.
///   4. Completeness — every collocation in the stream appears in the output.
pub fn extract_all(tags: &Vec<POS>) -> (result: Vec<Collocation>)
    ensures
        all_entries_valid(result@, tags@),
        positions_ascending(result@),
        forall|i: int| 0 <= i && i + 1 < tags@.len()
            && is_collocation_at(tags@, i) ==>
            exists|k: int| #![trigger result@[k]] 0 <= k < result@.len()
                && result@[k].pos == i as usize,
{
    let mut out: Vec<Collocation> = Vec::new();
    let len = tags.len();
    if len < 2 {
        return out;
    }
    let mut i: usize = 0;
    while i + 1 < len
        invariant
            0 <= i <= len - 1,
            len == tags@.len(),
            len >= 2,
            all_entries_valid(out@, tags@),
            positions_ascending(out@),
            all_positions_below(out@, i as int),
            complete_up_to(out@, tags@, i as int),
        decreases len - i,
    {
        let t0: POS = tags[i];
        let t1: POS = tags[i + 1];

        // The exec match and spec predicate coincide.
        proof {
            assert(t0 == tags@[i as int]);
            assert(t1 == tags@[i as int + 1]);
        }

        if (t0 == POS::Adj  && t1 == POS::Noun)
            || (t0 == POS::Verb && t1 == POS::Noun)
            || (t0 == POS::Noun && t1 == POS::Verb)
        {
            let pat: Pattern =
                if t0 == POS::Adj {
                    Pattern::AdjNoun
                } else if t0 == POS::Verb {
                    Pattern::VerbNoun
                } else {
                    Pattern::NounVerb
                };
            let c = Collocation { pattern: pat, pos: i };
            proof {
                assert(is_collocation_at(tags@, i as int));
                assert(c.pattern == pattern_at(tags@, i as int));
                // New entry's pos == i, which is > all existing positions
                let ghost old_out = out@;
                let ghost new_out = old_out.push(c);
                assert forall|k: int, l: int|
                    #![trigger new_out[k], new_out[l]]
                    0 <= k < l < new_out.len()
                    implies new_out[k].pos < new_out[l].pos
                by {
                    if l < old_out.len() as int {
                        assert(new_out[k] == old_out[k]);
                        assert(new_out[l] == old_out[l]);
                    } else {
                        assert(new_out[l] == c);
                        if k < old_out.len() as int {
                            assert(new_out[k] == old_out[k]);
                        }
                    }
                }
                // Completeness: the new entry covers position i
                assert forall|j: int| 0 <= j < (i + 1) as int && j + 1 < tags@.len()
                    && is_collocation_at(tags@, j)
                    implies exists|kk: int| #![trigger new_out[kk]] 0 <= kk < new_out.len()
                        && new_out[kk].pos == j as usize
                by {
                    if j < i as int {
                        // From previous complete_up_to invariant, there
                        // exists k with old_out[k].pos == j. That k still
                        // works in new_out since new_out[k] == old_out[k]
                        // for k < old_out.len().
                        let kk = choose|k: int| #![trigger old_out[k]]
                            0 <= k < old_out.len() && old_out[k].pos == j as usize;
                        assert(new_out[kk] == old_out[kk]);
                    } else {
                        // j == i; the witness is the new entry
                        let kk = old_out.len() as int;
                        assert(new_out[kk] == c);
                        assert(new_out[kk].pos == i);
                    }
                }
            }
            out.push(c);
        } else {
            proof {
                // Not a collocation at i, so complete_up_to(i+1) follows
                // directly from complete_up_to(i).
                assert(!is_collocation_at(tags@, i as int));
            }
        }

        i = i + 1;
    }
    out
}

} // verus!
