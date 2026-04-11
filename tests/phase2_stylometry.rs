use writer_cli::corpus::sample::{Sample, SampleMetadata};
use writer_cli::stylometry::ai_slop;
use writer_cli::stylometry::features::function_words;
use writer_cli::stylometry::features::lengths;
use writer_cli::stylometry::features::ngrams;
use writer_cli::stylometry::features::punctuation::PunctuationStats;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;
use writer_cli::stylometry::scoring;

const HUMAN_TEXT: &str = "I walked to the store yesterday. The rain was light, barely a drizzle. \
I picked up bread, milk, and some oranges. On the way back I noticed the old oak tree had lost \
a branch — probably from the wind last night. The neighbourhood cats were out, as usual, lounging \
on the warm pavement. I thought about calling my mother but decided to wait until evening.";

const AI_TEXT: &str = "In today's rapidly evolving landscape, it's worth noting that the multifaceted \
nature of modern commerce necessitates a holistic approach. Furthermore, stakeholders must leverage \
cutting-edge frameworks to navigate the intricate tapestry of consumer behavior. This paradigm shift \
underscores the paramount importance of robust, comprehensive strategies that foster synergy across \
the ecosystem. Delving deeper into these dynamics reveals a transformative opportunity to streamline \
operations and catalyze unprecedented growth.";

#[test]
fn length_stats_produce_reasonable_values() {
    let stats = lengths::word_lengths(HUMAN_TEXT);
    assert!(stats.mean > 2.0 && stats.mean < 8.0, "mean word length: {}", stats.mean);
    assert!(stats.sd > 0.0);
}

#[test]
fn sentence_lengths_differ_between_styles() {
    let human = lengths::sentence_lengths(HUMAN_TEXT);
    let ai = lengths::sentence_lengths(AI_TEXT);
    // AI text typically has longer sentences
    assert!(ai.mean > human.mean * 0.5, "AI mean {} vs human mean {}", ai.mean, human.mean);
}

#[test]
fn function_words_detected() {
    let fw = function_words::compute(HUMAN_TEXT);
    assert!(fw.contains_key("the"), "should detect 'the'");
    assert!(fw.contains_key("i"), "should detect 'i'");
    assert!(fw.len() > 5, "should find multiple function words");
}

#[test]
fn trigrams_produce_results() {
    let tris = ngrams::trigrams(HUMAN_TEXT);
    assert!(!tris.is_empty());
    // "the" should be a common trigram
    assert!(tris.iter().any(|(g, _)| g == "the"), "expected 'the' trigram");
}

#[test]
fn punctuation_detects_em_dashes() {
    let text_with_dashes = "Hello — world — test. More text; here: too!";
    let stats = PunctuationStats::compute(text_with_dashes);
    assert!(stats.em_dashes_per_1k > 0.0, "should detect em dashes");
    assert!(stats.semicolons_per_1k > 0.0, "should detect semicolons");
    assert!(stats.exclamations_per_1k > 0.0, "should detect exclamations");
}

#[test]
fn ai_slop_lists_are_populated() {
    assert!(ai_slop::BANNED_WORDS.len() >= 50);
    assert!(ai_slop::BANNED_PHRASES.len() >= 10);
    // No duplicates
    let word_set: std::collections::HashSet<_> = ai_slop::BANNED_WORDS.iter().collect();
    assert_eq!(word_set.len(), ai_slop::BANNED_WORDS.len(), "no duplicate banned words");
}

#[test]
fn fingerprint_computes_from_samples() {
    let samples: Vec<Sample> = vec![
        Sample::new(HUMAN_TEXT.into(), SampleMetadata::default()),
    ];
    let fp = StylometricFingerprint::compute(&samples);
    assert!(fp.word_count > 0);
    assert!(fp.sentence_length.mean > 0.0);
    assert!(!fp.function_words.is_empty());
    assert!(!fp.ngram_profile.is_empty());
}

#[test]
fn fingerprint_serializes_roundtrip() {
    let samples = vec![Sample::new(HUMAN_TEXT.into(), SampleMetadata::default())];
    let fp = StylometricFingerprint::compute(&samples);
    let json = serde_json::to_string(&fp).unwrap();
    let deserialized: StylometricFingerprint = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.word_count, fp.word_count);
}

#[test]
fn scoring_human_text_closer_to_human_fingerprint() {
    let samples = vec![Sample::new(HUMAN_TEXT.into(), SampleMetadata::default())];
    let fp = StylometricFingerprint::compute(&samples);

    let human_distance = scoring::distance(HUMAN_TEXT, &fp);
    let ai_distance = scoring::distance(AI_TEXT, &fp);

    assert!(
        human_distance.overall < ai_distance.overall,
        "Human text ({:.3}) should be closer to human fingerprint than AI text ({:.3})",
        human_distance.overall,
        ai_distance.overall
    );
}

#[test]
fn scoring_ai_text_has_higher_slop_penalty() {
    let samples = vec![Sample::new(HUMAN_TEXT.into(), SampleMetadata::default())];
    let fp = StylometricFingerprint::compute(&samples);

    let human_report = scoring::distance(HUMAN_TEXT, &fp);
    let ai_report = scoring::distance(AI_TEXT, &fp);

    assert!(
        ai_report.slop_score > human_report.slop_score,
        "AI text slop ({:.3}) should be higher than human ({:.3})",
        ai_report.slop_score,
        human_report.slop_score
    );
}

#[test]
fn sentence_segmentation_audit() {
    use writer_cli::stylometry::features::lengths;
    // Test with abbreviations that should NOT split
    let text_with_abbrevs = "Mr. Ford Prefect arrived at 3 p.m. on Tuesday. He was quite annoyed.";
    let stats = lengths::sentence_lengths(text_with_abbrevs);
    println!("Abbrev text: mean={:.1} sd={:.1} sentences detected", stats.mean, stats.sd);

    // Test with Adams-like short fragments
    let adams_like = "Don't panic. Mostly harmless. So long, and thanks for all the fish! The ships hung in the sky in much the same way that bricks don't.";
    let stats2 = lengths::sentence_lengths(adams_like);
    println!("Adams-like: mean={:.1} sd={:.1}", stats2.mean, stats2.sd);
}
