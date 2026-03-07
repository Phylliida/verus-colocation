use std::{fs::read_to_string, collections::HashMap, fmt::Write};
use serde_json as json;
use serde::Serialize;

/// Return the last `n` characters of `s`, or `""` if `s` has fewer than `n` chars.
fn suffix(s: &str, n: usize) -> &str {
    let char_count = s.chars().count();
    if char_count <= n {
        return "";
    }
    let byte_offset = s.char_indices()
        .nth(char_count - n)
        .map(|(i, _)| i)
        .unwrap_or(0);
    &s[byte_offset..]
}

struct AveragedPerceptron {
    feature_weights: HashMap<String, HashMap<String, f32>>,
    classes: Vec<String>,
}

impl AveragedPerceptron {

    pub fn new(
        weights_filepath: &str,
        classes_filepath: &str,
    ) -> AveragedPerceptron {
        let mut feature_weights: HashMap<String, HashMap<String, f32>> = HashMap::new();
        let weights_str: String = read_to_string(weights_filepath).expect("Could not read weights.json");
        let weights_json: json::Value = json::from_str(weights_str.as_str()).expect("Could not convert weights.json to json::Value");
        for (feature_name, value) in weights_json.as_object().unwrap() {
            let mut weights: HashMap<String, f32> = HashMap::new();
            for (tag, weight) in value.as_object().unwrap() {
                weights.insert(tag.to_string(), weight.as_f64().unwrap() as f32);
            }
            feature_weights.insert(feature_name.to_string(), weights);
        }

        let classes_str = read_to_string(classes_filepath).expect("Could not read classes.txt as string");
        let classes: Vec<String> = classes_str.split('\n')
            .map(|class| class.trim().to_string())
            .collect();

        AveragedPerceptron { feature_weights, classes }
    }

    /// Accumulate weights for a single feature into `scores`.
    #[inline]
    fn accumulate<'a>(&'a self, scores: &mut HashMap<&'a str, f32>, feature: &str) {
        if let Some(weights) = self.feature_weights.get(feature) {
            for (label, weight) in weights {
                *scores.entry(label.as_str()).or_insert(0.0) += weight;
            }
        }
    }

    /// Predict POS tag given contextual features.
    ///
    /// Uses reusable `scores` and `buf` buffers to avoid per-word allocations.
    /// Softmax is skipped — we only need the argmax class.
    pub fn predict_in_context<'a>(
        &'a self,
        scores: &mut HashMap<&'a str, f32>,
        buf: &mut String,
        word: &str,
        context_i: &str,
        context_im1: &str,
        context_im2: &str,
        context_ip1: &str,
        context_ip2: &str,
        prev: &str,
        prev2: &str,
    ) -> &'a str {
        scores.clear();

        // bias (static key, no formatting needed)
        self.accumulate(scores, "bias");

        macro_rules! feat {
            ($($arg:tt)*) => {{
                buf.clear();
                write!(buf, $($arg)*).unwrap();
                self.accumulate(scores, buf);
            }};
        }

        feat!("i suffix {}", suffix(word, 3));
        if let Some(c) = word.chars().nth(1) {
            feat!("i pref1 {}", c);
        } else {
            feat!("i pref1 ");
        }
        feat!("i-1 tag {}", prev);
        feat!("i-2 tag {}", prev2);
        feat!("i tag+i-2 tag {} {}", prev, prev2);
        feat!("i word {}", context_i);
        feat!("i-1 tag+i word {} {}", prev, context_i);
        feat!("i-1 word {}", context_im1);
        feat!("i-2 word {}", context_im2);
        feat!("i+1 word {}", context_ip1);
        feat!("i+2 word {}", context_ip2);
        feat!("i+1 suffix {}", suffix(context_ip1, 3));
        feat!("i-1 suffix {}", suffix(context_im1, 3));

        self.classes.iter()
            .max_by(|a, b| {
                scores.get(a.as_str()).unwrap_or(&0.0)
                    .partial_cmp(scores.get(b.as_str()).unwrap_or(&0.0))
                    .unwrap()
            })
            .unwrap()
    }

}

#[derive(Serialize)]
pub struct Tag<'a>{
    pub word: &'a str,
    pub tag: String,
    pub conf: f32
}

pub struct PerceptronTagger {
    model: AveragedPerceptron,
    tags: HashMap<String, String>
}

impl PerceptronTagger {

    #[must_use]
    pub fn new(
        weights_filepath: &str,
        classes_filepath: &str,
        tags_filepath: &str
    ) -> PerceptronTagger {
        let mut tags: HashMap<String, String> = HashMap::new();
        let tags_str: String = read_to_string(tags_filepath).expect("Could not read tags.json");
        let tags_json: json::Value = json::from_str(&tags_str).expect("Could not convert tags.json to json::Value");
        for (word, tag) in tags_json.as_object().unwrap() {
            tags.insert(word.as_str().to_owned(), tag.as_str().unwrap().to_owned());
        }

        PerceptronTagger {
            model: AveragedPerceptron::new(weights_filepath, classes_filepath),
            tags
        }
    }

    pub fn tag<'a>(
        &'a self,
        sentence: &'a str
    ) -> Vec<Tag> {
        self.assign_tags(sentence.split_whitespace().collect::<Vec<&str>>())
    }

    /// Look up the POS tag for a known word (from tags.json).
    /// Returns None if the word requires model prediction.
    pub fn lookup_tag(&self, word: &str) -> Option<&str> {
        self.tags.get(word).map(|s| s.as_str())
    }

    fn assign_tags<'a>(
        &'a self,
        tokens: Vec<&'a str>
    ) -> Vec<Tag> {
        let mut prev: &str = "-START-";
        let mut prev2: &str = "-START2-";
        let mut output: Vec<Tag> = Vec::with_capacity(tokens.len());

        // Reusable buffers for predict_in_context
        let mut scores: HashMap<&str, f32> = HashMap::new();
        let mut buf = String::with_capacity(64);

        let mut context: Vec<&str> = Vec::with_capacity(tokens.len() + 4);
        context.push("-START-");
        context.push("-START2-");
        for token in &tokens {
            context.push(
                if token.contains("'-'") && !token.starts_with('-') {
                    "!HYPHEN"
                }
                else if token.parse::<usize>().is_ok() && token.len() == 4 {
                    "!YEAR"
                }
                else if token.starts_with(|c: char| c.is_ascii_digit()) {
                    "!DIGITS"
                }
                else {
                    token
                }
            );
        }
        context.push("-END-");
        context.push("-END2-");

        for (i, token) in tokens.into_iter().enumerate() {
            if let Some(known_tag) = self.tags.get(token) {
                output.push(
                    Tag {
                        word: token,
                        tag: known_tag.clone(),
                        conf: 0.0
                    }
                );
                prev2 = prev;
                prev = known_tag;
            }
            else {
                let ci = i + 2;
                let tag = self.model.predict_in_context(
                    &mut scores,
                    &mut buf,
                    token,
                    context[ci],
                    context[ci - 1],
                    context[ci - 2],
                    context[ci + 1],
                    context[ci + 2],
                    prev,
                    prev2,
                );
                output.push(
                    Tag {
                        word: token,
                        tag: tag.to_string(),
                        conf: 0.0
                    }
                );
                prev2 = prev;
                prev = tag;
            }
        }

        output
    }

}
