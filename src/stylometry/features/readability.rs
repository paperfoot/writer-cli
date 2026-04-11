//! Readability metrics — evidence-based complexity signals.
//!
//! References:
//! - Flesch, R. (1948). "A new readability yardstick." Journal of Applied Psychology.
//! - Coleman, M. & Liau, T.L. (1975). "A computer readability formula." Journal of Applied Psychology.
//! - Senter, R.J. & Smith, E.A. (1967). "Automated Readability Index." AMRL-TR-66-220.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadabilityStats {
    /// Flesch-Kincaid Grade Level (higher = more complex)
    pub flesch_kincaid_grade: f64,
    /// Coleman-Liau Index
    pub coleman_liau_index: f64,
    /// Automated Readability Index
    pub ari: f64,
    /// Average syllables per word
    pub avg_syllables_per_word: f64,
}

impl ReadabilityStats {
    pub fn compute(text: &str) -> Self {
        let words: Vec<&str> = text.unicode_words().collect();
        let sentences = super::sentences::split_sentences(text).len() as f64;
        let word_count = words.len() as f64;

        if word_count == 0.0 || sentences == 0.0 {
            return Self::default();
        }

        let total_syllables: f64 = words.iter().map(|w| count_syllables(w) as f64).sum();
        let total_chars: f64 = words.iter().map(|w| w.chars().count() as f64).sum();

        let avg_syllables = total_syllables / word_count;
        let words_per_sentence = word_count / sentences;

        // Flesch-Kincaid Grade Level
        let fk_grade = 0.39 * words_per_sentence + 11.8 * avg_syllables - 15.59;

        // Coleman-Liau Index
        let l = (total_chars / word_count) * 100.0; // avg letters per 100 words
        let s = (sentences / word_count) * 100.0; // avg sentences per 100 words
        let cli = 0.0588 * l - 0.296 * s - 15.8;

        // Automated Readability Index
        let chars_per_word = total_chars / word_count;
        let ari = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43;

        Self {
            flesch_kincaid_grade: fk_grade,
            coleman_liau_index: cli,
            ari,
            avg_syllables_per_word: avg_syllables,
        }
    }
}

/// Estimate syllable count using the vowel-group method.
/// Not perfect but fast and sufficient for stylometric comparison.
fn count_syllables(word: &str) -> usize {
    let word = word.to_lowercase();
    if word.len() <= 3 {
        return 1;
    }

    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut count = 0;
    let mut prev_vowel = false;

    for ch in word.chars() {
        let is_vowel = vowels.contains(&ch);
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }

    // Silent e
    if word.ends_with('e') && count > 1 {
        count -= 1;
    }

    count.max(1)
}
