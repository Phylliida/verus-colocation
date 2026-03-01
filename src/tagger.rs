//! POS tagger via spaCy (Python) through pyo3 bindings.
//!
//! This module is exec-only and not verified by Verus.  It sits outside the
//! verification boundary: spaCy is treated as an unverified oracle.
//!
//! Enable with `cargo build --features tagger`.

use pyo3::prelude::*;

/// Simplified POS tag matching the verified `colocation::POS` enum.
///
/// This is the exec-side mirror.  The mapping to the verified enum happens
/// at the boundary when feeding tags into `extract_all`.
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

/// Map spaCy's Universal POS tag string to our enum.
fn map_pos(tag: &str) -> POS {
    match tag {
        "ADJ" => POS::Adj,
        "NOUN" | "PROPN" => POS::Noun,
        "VERB" | "AUX" => POS::Verb,
        "ADV" => POS::Adv,
        "ADP" => POS::Prep,
        "DET" => POS::Det,
        _ => POS::Other,
    }
}

/// A tagged token: surface form + POS tag.
#[derive(Debug, Clone)]
pub struct TaggedToken {
    pub word: String,
    pub pos: POS,
}

/// Holds a loaded spaCy model across calls.
pub struct SpacyTagger {
    nlp: Py<PyAny>,
}

impl SpacyTagger {
    /// Load a spaCy model by name.
    ///
    /// Common models: `"en_core_web_sm"`, `"en_core_web_md"`,
    /// `"en_core_web_lg"`, `"en_core_web_trf"`.
    ///
    /// The Python interpreter is auto-initialized on first call.
    pub fn new(model_name: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let spacy = py.import("spacy")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("disable", vec!["parser", "ner", "lemmatizer"])?;
            let nlp = spacy.call_method("load", (model_name,), Some(&kwargs))?;
            Ok(SpacyTagger {
                nlp: nlp.unbind(),
            })
        })
    }

    /// POS-tag a text string.  Returns tagged tokens.
    pub fn tag(&self, text: &str) -> PyResult<Vec<TaggedToken>> {
        Python::with_gil(|py| {
            let nlp = self.nlp.bind(py);
            let doc = nlp.call1((text,))?;
            let mut result = Vec::new();
            for token in doc.try_iter()? {
                let token = token?;
                let word: String = token.getattr("text")?.extract()?;
                let pos_str: String = token.getattr("pos_")?.extract()?;
                result.push(TaggedToken {
                    word,
                    pos: map_pos(&pos_str),
                });
            }
            Ok(result)
        })
    }

    /// Return only POS tags (dropping surface forms), ready for the verified
    /// collocation extractor.
    pub fn tag_pos_only(&self, text: &str) -> PyResult<Vec<POS>> {
        Ok(self.tag(text)?.into_iter().map(|t| t.pos).collect())
    }

    /// Batch POS-tag multiple texts via `nlp.pipe()`.
    /// Returns one `Vec<TaggedToken>` per input text, in the same order.
    pub fn tag_batch(&self, texts: &[&str], batch_size: usize) -> PyResult<Vec<Vec<TaggedToken>>> {
        Python::with_gil(|py| {
            let nlp = self.nlp.bind(py);
            let pipe_kwargs = pyo3::types::PyDict::new(py);
            pipe_kwargs.set_item("batch_size", batch_size)?;
            let docs = nlp.call_method("pipe", (texts.to_vec(),), Some(&pipe_kwargs))?;
            let mut results = Vec::with_capacity(texts.len());
            for doc in docs.try_iter()? {
                py.check_signals()?; // allow Ctrl+C between docs
                let doc = doc?;
                let mut tokens = Vec::new();
                for token in doc.try_iter()? {
                    let token = token?;
                    let word: String = token.getattr("text")?.extract()?;
                    let pos_str: String = token.getattr("pos_")?.extract()?;
                    tokens.push(TaggedToken {
                        word,
                        pos: map_pos(&pos_str),
                    });
                }
                results.push(tokens);
            }
            Ok(results)
        })
    }
}
