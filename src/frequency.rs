use vstd::prelude::*;
use crate::colocation::Pattern;

verus! {

/// A collocation together with its observed frequency.
#[derive(Copy, Clone)]
pub struct RankedEntry {
    pub pattern: Pattern,
    pub pos: usize,
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Specification: sorted (descending by count)
// ---------------------------------------------------------------------------

/// Adjacent-pair formulation of descending sort.
pub open spec fn is_pairwise_descending(entries: Seq<RankedEntry>) -> bool {
    forall|i: int| #![trigger entries[i], entries[i + 1]]
        0 <= i && i + 1 < entries.len() ==> entries[i].count >= entries[i + 1].count
}

// ---------------------------------------------------------------------------
// Specification: threshold filtering
// ---------------------------------------------------------------------------

/// True when every entry has count >= `min_freq`.
pub open spec fn all_above_threshold(entries: Seq<RankedEntry>, min_freq: u64) -> bool {
    forall|i: int| #![trigger entries[i]]
        0 <= i < entries.len() ==> entries[i].count >= min_freq
}

// ---------------------------------------------------------------------------
// Proof: pairwise ⇒ global sorted order
// ---------------------------------------------------------------------------

/// Pairwise descending implies the global sorted property.
pub proof fn lemma_pairwise_implies_sorted(s: Seq<RankedEntry>, i: int, j: int)
    requires
        is_pairwise_descending(s),
        0 <= i,
        i < j,
        j < s.len(),
    ensures
        s[i].count >= s[j].count,
    decreases j - i,
{
    let ghost _a = s[j - 1];
    let ghost _b = s[(j - 1) + 1];
    assert(s[j - 1].count >= s[j].count);

    if j == i + 1 {
    } else {
        lemma_pairwise_implies_sorted(s, i, j - 1);
    }
}

// ---------------------------------------------------------------------------
// Specification helpers for selection sort
// ---------------------------------------------------------------------------

/// arr[0..bound] is pairwise sorted descending.
pub open spec fn prefix_sorted(arr: Seq<RankedEntry>, bound: int) -> bool {
    forall|a: int, b: int|
        #![trigger arr[a], arr[b]]
        0 <= a < b < bound && b < arr.len() ==> arr[a].count >= arr[b].count
}

/// Everything in arr[0..pivot] is >= everything in arr[pivot..n].
pub open spec fn partitioned_at(arr: Seq<RankedEntry>, pivot: int) -> bool {
    forall|a: int, b: int|
        #![trigger arr[a], arr[b]]
        0 <= a < pivot && pivot <= b < arr.len() ==> arr[a].count >= arr[b].count
}

// ---------------------------------------------------------------------------
// Executable: selection sort (descending by count)
// ---------------------------------------------------------------------------

/// Sort `entries` in descending order by count.
///
/// Verified properties:
///   1. Result has same length as input.
///   2. Result is pairwise descending.
pub fn sort_descending(entries: &Vec<RankedEntry>) -> (result: Vec<RankedEntry>)
    ensures
        result@.len() == entries@.len(),
        is_pairwise_descending(result@),
{
    let mut arr: Vec<RankedEntry> = Vec::new();
    let mut ci: usize = 0;
    while ci < entries.len()
        invariant
            0 <= ci <= entries@.len(),
            arr@.len() == ci,
        decreases entries@.len() - ci,
    {
        arr.push(entries[ci]);
        ci = ci + 1;
    }

    let n = arr.len();
    if n <= 1 {
        return arr;
    }

    let mut i: usize = 0;
    while i + 1 < n
        invariant
            0 <= i < n,
            n == arr@.len(),
            n == entries@.len(),
            prefix_sorted(arr@, i as int),
            partitioned_at(arr@, i as int),
        decreases n - i,
    {
        // Find index of maximum in arr[i..]
        let mut max_idx: usize = i;
        let mut j: usize = i + 1;
        while j < n
            invariant
                i < j,
                j <= n,
                i <= max_idx < j,
                n == arr@.len(),
                forall|k: int| #![trigger arr@[k]]
                    i <= k < j ==> arr@[max_idx as int].count >= arr@[k].count,
            decreases n - j,
        {
            if arr[j].count > arr[max_idx].count {
                max_idx = j;
            }
            j = j + 1;
        }

        // Ghost snapshot before swap
        let ghost old_arr = arr@;
        let ghost max_val = arr@[max_idx as int];
        let ghost i_val = arr@[i as int];

        // max_val.count >= everything in old_arr[i..n]
        // (from the find-max loop postcondition)

        // Swap arr[i] and arr[max_idx]
        if max_idx != i {
            let tmp = arr[i];
            arr.set(i, arr[max_idx]);
            arr.set(max_idx, tmp);
        }

        proof {
            // After swap, arr@ relates to old_arr as:
            //   arr[i] == old max_val
            //   arr[max_idx] == old i_val
            //   arr[k] == old_arr[k]  for k != i, k != max_idx

            // GOAL 1: prefix_sorted(arr@, (i+1) as int)
            // Need: forall a < b < i+1 with b < n => arr[a].count >= arr[b].count
            assert forall|a: int, b: int|
                #![trigger arr@[a], arr@[b]]
                0 <= a < b < (i + 1) as int && b < n as int
                implies arr@[a].count >= arr@[b].count
            by {
                if b < i as int {
                    // Both a, b < i. Neither was swapped (swap only touches i and max_idx >= i).
                    // So arr[a] == old_arr[a] and arr[b] == old_arr[b].
                    // prefix_sorted(old_arr, i) gives old_arr[a] >= old_arr[b].
                    assert(arr@[a] == old_arr[a]);
                    assert(arr@[b] == old_arr[b]);
                } else {
                    // b == i (since b < i+1 and b >= i)
                    // arr[i] == max_val
                    // arr[a] with a < i: arr[a] == old_arr[a] (unchanged)
                    // Need: old_arr[a].count >= max_val.count
                    // From partitioned_at(old_arr, i): old_arr[a] >= old_arr[b'] for any b' in [i,n)
                    // In particular old_arr[a] >= old_arr[max_idx] == max_val
                    assert(arr@[a] == old_arr[a]);
                    assert(arr@[b] == max_val);
                    // old_arr[a] >= old_arr[max_idx] from partitioned_at
                    assert(old_arr[a].count >= old_arr[max_idx as int].count);
                }
            }

            // GOAL 2: partitioned_at(arr@, (i+1) as int)
            // Need: forall a < i+1, b in [i+1, n) => arr[a].count >= arr[b].count
            assert forall|a: int, b: int|
                #![trigger arr@[a], arr@[b]]
                0 <= a < (i + 1) as int && (i + 1) as int <= b < n as int
                implies arr@[a].count >= arr@[b].count
            by {
                // What is arr[b]?
                // If b == max_idx: arr[b] == i_val == old_arr[i]
                // Else: arr[b] == old_arr[b]
                // In either case, arr[b] was in old_arr[i..n],
                // so max_val.count >= arr[b].count.

                // What is arr[a]?
                if a < i as int {
                    // arr[a] == old_arr[a], and partitioned_at(old_arr, i) gives
                    // old_arr[a] >= old_arr[anything in [i,n)]
                    assert(arr@[a] == old_arr[a]);
                    if b == max_idx as int {
                        assert(arr@[b] == i_val);
                        // old_arr[a] >= old_arr[i] = i_val from partitioned_at
                        assert(old_arr[a].count >= old_arr[i as int].count);
                    } else {
                        assert(arr@[b] == old_arr[b]);
                        assert(old_arr[a].count >= old_arr[b].count);
                    }
                } else {
                    // a == i, so arr[a] == max_val
                    assert(arr@[a] == max_val);
                    // max_val.count >= everything in old_arr[i..n]
                    if b == max_idx as int {
                        assert(arr@[b] == i_val);
                        // max_val >= old_arr[i] = i_val
                        assert(max_val.count >= old_arr[i as int].count);
                    } else {
                        assert(arr@[b] == old_arr[b]);
                        assert(max_val.count >= old_arr[b].count);
                    }
                }
            }
        }

        i = i + 1;
    }

    proof {
        assert forall|a: int|
            #![trigger arr@[a], arr@[a + 1]]
            0 <= a && a + 1 < n as int
            implies arr@[a].count >= arr@[a + 1].count
        by {
            if a + 1 < i as int {
                // Both in prefix_sorted region
            } else {
                // a < i <= n-1, a+1 >= i  →  use partitioned_at
            }
        }
    }

    arr
}

// ---------------------------------------------------------------------------
// Executable: threshold filtering
// ---------------------------------------------------------------------------

/// Filter `entries` to only those with `count >= min_freq`.
///
/// Verified properties:
///   1. Every entry in the result has count >= min_freq.
///   2. Result length <= input length.
pub fn filter_by_threshold(
    entries: &Vec<RankedEntry>,
    min_freq: u64,
) -> (result: Vec<RankedEntry>)
    ensures
        all_above_threshold(result@, min_freq),
        result@.len() <= entries@.len(),
{
    let mut out: Vec<RankedEntry> = Vec::new();
    let mut i: usize = 0;
    while i < entries.len()
        invariant
            0 <= i <= entries@.len(),
            all_above_threshold(out@, min_freq),
            out@.len() <= i,
        decreases entries@.len() - i,
    {
        if entries[i].count >= min_freq {
            out.push(entries[i]);
        }
        i = i + 1;
    }
    out
}

} // verus!
