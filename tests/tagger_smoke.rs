#![cfg(feature = "tagger")]

use verus_colocation::tagger::{SpacyTagger, POS};

#[test]
fn smoke_test_spacy_tagger() {
    let tagger = SpacyTagger::new("en_core_web_sm")
        .expect("failed to load en_core_web_sm — is it installed?");

    let tokens = tagger
        .tag("busy bees buzz loudly")
        .expect("tagging failed");

    assert_eq!(tokens.len(), 4);
    assert_eq!(tokens[0].word, "busy");
    assert_eq!(tokens[0].pos, POS::Adj);
    assert_eq!(tokens[1].word, "bees");
    assert_eq!(tokens[1].pos, POS::Noun);
    assert_eq!(tokens[2].word, "buzz");
    assert_eq!(tokens[2].pos, POS::Verb);
    assert_eq!(tokens[3].word, "loudly");
    assert_eq!(tokens[3].pos, POS::Adv);
}

#[test]
fn pos_only_returns_just_tags() {
    let tagger = SpacyTagger::new("en_core_web_sm")
        .expect("failed to load en_core_web_sm");

    let tags = tagger
        .tag_pos_only("The quick brown fox")
        .expect("tagging failed");

    assert_eq!(tags.len(), 4);
    assert_eq!(tags[0], POS::Det); // The
    assert_eq!(tags[1], POS::Adj); // quick
    assert_eq!(tags[2], POS::Adj); // brown
    assert_eq!(tags[3], POS::Noun); // fox
}
