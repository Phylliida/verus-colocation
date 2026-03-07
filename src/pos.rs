//! Shared POS types and tagger trait.
//!
//! Always compiled (no feature gate). Both the spaCy backend (`tagger.rs`) and
//! the pure-Rust backend (`postagger_backend.rs`) import from here.

/// Simplified POS tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum POS {
    Adj,
    Noun,
    Verb,
    Adv,
    Prep,
    Det,
    Other,
}

/// A tagged token: surface form + POS tag.
#[derive(Debug, Clone)]
pub struct TaggedToken {
    pub word: String,
    pub pos: POS,
}

/// Backend-agnostic POS tagger.
pub trait Tagger {
    fn tag_batch(
        &self,
        texts: &[&str],
        batch_size: usize,
    ) -> Result<Vec<Vec<TaggedToken>>, Box<dyn std::error::Error>>;

    /// Look up POS for a known word without running the full tagger.
    /// Returns None if the word is unknown and requires model prediction.
    fn lookup_pos(&self, _word: &str) -> Option<POS> {
        None
    }
}
