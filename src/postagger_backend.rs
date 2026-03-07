//! Pure-Rust POS tagger backend using the `postagger` crate.
//!
//! This is an alternative to the spaCy/Python backend that requires no
//! external runtime. It uses an averaged perceptron model ported from NLTK.
//!
//! Enable with `cargo build --features postagger-backend`.

use crate::pos::{Tagger, TaggedToken, POS};

/// Map Penn Treebank tags (output by `postagger`) to our `POS` enum.
fn map_penn(tag: &str) -> POS {
    match tag {
        "JJ" | "JJR" | "JJS" => POS::Adj,
        "NN" | "NNS" | "NNP" | "NNPS" => POS::Noun,
        "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" => POS::Verb,
        "RB" | "RBR" | "RBS" => POS::Adv,
        "IN" => POS::Prep,
        "DT" | "PDT" | "WDT" => POS::Det,
        _ => POS::Other,
    }
}

/// Pure-Rust POS tagger wrapping `postagger::PerceptronTagger`.
pub struct RustTagger {
    inner: postagger::PerceptronTagger,
}

impl RustTagger {
    /// Create a new `RustTagger` from model files on disk.
    ///
    /// * `weights_path` – path to `weights.json`
    /// * `classes_path` – path to `classes.txt`
    /// * `tags_path`    – path to `tags.json`
    pub fn new(weights_path: &str, classes_path: &str, tags_path: &str) -> Self {
        RustTagger {
            inner: postagger::PerceptronTagger::new(weights_path, classes_path, tags_path),
        }
    }
}

impl Tagger for RustTagger {
    fn tag_batch(
        &self,
        texts: &[&str],
        _batch_size: usize,
    ) -> Result<Vec<Vec<TaggedToken>>, Box<dyn std::error::Error>> {
        let mut results = Vec::with_capacity(texts.len());
        for &text in texts {
            let tags = self.inner.tag(text);
            let tokens: Vec<TaggedToken> = tags
                .into_iter()
                .map(|t| TaggedToken {
                    word: t.word.to_string(),
                    pos: map_penn(&t.tag),
                })
                .collect();
            results.push(tokens);
        }
        Ok(results)
    }
}
