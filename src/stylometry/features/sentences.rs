//! Rule-based sentence segmentation.
//!
//! Replaces `unicode_segmentation::unicode_sentences()` which splits on
//! abbreviation periods (Mr., p.m., etc.), inflating sentence counts.
//!
//! Forward-scanning approach: find candidate terminal punctuation,
//! consume trailing closers, then veto the split if the context indicates
//! an abbreviation, initialism, or decimal number.

/// Strong no-break: titles/honorifics — almost never end a sentence.
const STRONG_ABBREVS: &[&str] = &[
    "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "fr", "sr", "jr", "capt", "lt", "col", "gen",
    "sgt", "cpl", "pvt", "st", "mt", "messrs", "mme", "mlle",
];

/// Usually internal: Latin/scholarly — rarely sentence-terminal.
const INTERNAL_ABBREVS: &[&str] = &[
    "eg", "ie", "cf", "viz", "approx", "dept", "govt", "no", "nos", "vol", "vols", "pp", "ch",
    "fig", "figs", "sec", "ed", "eds", "inc", "ltd", "corp", "assn",
];

/// Weak: can end a sentence — split only when next token is uppercase.
const WEAK_ABBREVS: &[&str] = &["etc", "vs", "am", "pm"];

/// Split text into sentences using rule-based boundary detection.
pub fn split_sentences(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let len = chars.len();

    let mut sentences = Vec::new();
    let mut sent_start_byte = 0;
    let mut i = 0;

    while i < len {
        let (_, ch) = chars[i];

        if !is_terminal(ch) {
            i += 1;
            continue;
        }

        let terminal_char = ch;

        // Consume repeated terminal punctuation: ..., ?!, !!
        let mut end_i = i;
        while end_i + 1 < len && is_terminal(chars[end_i + 1].1) {
            end_i += 1;
        }
        let is_multi_dot = terminal_char == '.' && end_i > i;

        // Consume trailing closing punctuation: " ' ) ] }
        while end_i + 1 < len && is_closer(chars[end_i + 1].1) {
            end_i += 1;
        }

        // Byte position after the consumed cluster
        let split_byte = if end_i + 1 < len {
            chars[end_i + 1].0
        } else {
            text.len()
        };

        let should_split = if is_multi_dot {
            // Ellipsis: only split if followed by uppercase
            next_starts_uppercase(&chars, end_i + 1)
        } else if terminal_char == '.' {
            should_split_at_period(text, &chars, i, end_i)
        } else {
            // ! or ? — split unless followed by lowercase (dialog tag)
            // "Go!" he said. → don't split at !
            // "Go!" She went. → split at !
            should_split_at_bang_question(&chars, end_i)
        };

        if should_split {
            let sentence = text[sent_start_byte..split_byte].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            sent_start_byte = split_byte;
        }

        i = end_i + 1;
    }

    let tail = text[sent_start_byte..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

/// Decide whether a single period at char index `period_i` is a sentence boundary.
fn should_split_at_period(
    text: &str,
    chars: &[(usize, char)],
    period_i: usize,
    end_i: usize,
) -> bool {
    // Decimal: digit.digit
    if is_decimal_period(chars, period_i) {
        return false;
    }

    // Mid-sequence check: if the very next char is a letter followed by a period,
    // we're inside a dotted abbreviation like "e.g." or "H.G." — never split here.
    if is_mid_dotted_sequence(chars, period_i) {
        return false;
    }

    // Collect the full abbreviation group by walking backward: "p.m", "e.g", "B.B.C"
    let group = collect_dot_group(text, chars, period_i);

    if !group.is_empty() {
        // Normalize: strip dots, lowercase
        let normalized: String = group.to_lowercase().chars().filter(|c| *c != '.').collect();

        // Check abbreviation categories
        if let Some(kind) = classify_abbrev(&normalized) {
            return match kind {
                AbbrevKind::Strong | AbbrevKind::Internal => false,
                AbbrevKind::Weak => next_starts_uppercase(chars, end_i + 1),
            };
        }

        // Dotted initialism: all parts are single uppercase letters
        // e.g. "H.G", "B.B.C", "U.S.A"
        // Never split — false splits (1-word sentences) are worse for stylometry
        // than false merges. "H.G. Wells" must stay together.
        let parts: Vec<&str> = group.split('.').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2
            && parts
                .iter()
                .all(|p| p.len() == 1 && p.chars().all(|c| c.is_uppercase()))
        {
            return false;
        }
    }

    // Single letter before period: "H. G. Wells" or "p. 42"
    if period_i > 0 {
        let before = chars[period_i - 1].1;
        if before.is_alphabetic() {
            let is_single = period_i < 2
                || chars[period_i - 2].1.is_whitespace()
                || chars[period_i - 2].1 == '.';
            if is_single {
                return false;
            }
        }
    }

    // Default: it's a sentence boundary
    true
}

/// Check if we're in the middle of a dotted sequence: the char immediately after
/// this period is a letter, and shortly after that is another period.
/// Catches intermediate dots in "e.g.", "p.m.", "H.G.", "B.B.C." etc.
fn is_mid_dotted_sequence(chars: &[(usize, char)], period_i: usize) -> bool {
    let len = chars.len();
    // Next char must exist and be a letter (no whitespace gap)
    if period_i + 1 >= len {
        return false;
    }
    let next = chars[period_i + 1].1;
    if !next.is_alphabetic() {
        return false;
    }

    // After the letter, look for a period (possibly with more letters first for multi-char segments)
    let mut j = period_i + 2;
    // Allow a short letter run (1-5 chars) before the next period
    while j < len && chars[j].1.is_alphabetic() && j - (period_i + 1) <= 5 {
        j += 1;
    }
    j < len && chars[j].1 == '.'
}

/// For ! and ?: split unless followed by lowercase (dialog attribution).
/// "Go!" he said. → don't split at !
/// "Go!" She went. → split at !
fn should_split_at_bang_question(chars: &[(usize, char)], end_i: usize) -> bool {
    // Look at what follows
    for &(_, ch) in chars.get(end_i + 1..).unwrap_or(&[]) {
        if ch.is_whitespace() {
            continue;
        }
        // Lowercase start → likely dialog tag, don't split
        if ch.is_lowercase() {
            return false;
        }
        // Anything else (uppercase, quote, etc.) → split
        return true;
    }
    true // EOF → split
}

/// Walk backward from the period at `period_i` to collect a dot-group.
///
/// A dot-group is a sequence like "p.m", "e.g", "H.G", "B.B.C" — alternating
/// short letter sequences and dots leading up to the current period.
///
/// Returns the group as a string (without the final period), or "" if none found.
fn collect_dot_group<'a>(text: &'a str, chars: &[(usize, char)], period_i: usize) -> &'a str {
    if period_i == 0 {
        return "";
    }

    // Walk backward from just before the period
    let pos = period_i - 1;

    // The char just before the period should be a letter
    if !chars[pos].1.is_alphabetic() {
        return "";
    }

    // Walk backward: we expect patterns like letter(s).letter(s).letter(s)
    // Each segment is 1-5 letters followed by a dot
    let group_end_byte = chars[pos].0 + chars[pos].1.len_utf8();
    let mut group_start = pos;

    loop {
        // Skip backward through letters
        while group_start > 0 && chars[group_start - 1].1.is_alphabetic() {
            group_start -= 1;
        }

        // Check if there's a dot before these letters
        if group_start > 0 && chars[group_start - 1].1 == '.' {
            let dot_pos = group_start - 1;
            // And a letter before the dot?
            if dot_pos > 0 && chars[dot_pos - 1].1.is_alphabetic() {
                group_start = dot_pos - 1;
                // Continue walking backward
                continue;
            }
        }

        break;
    }

    // Check if what we found is actually preceded by whitespace or start-of-text
    // (to avoid matching the end of a regular word)
    if group_start > 0 && !chars[group_start - 1].1.is_whitespace() {
        // Not a standalone abbreviation — it's part of a longer word
        // But could still be something like "(e.g." with a paren
        if group_start > 0 && !is_opener(chars[group_start - 1].1) {
            return "";
        }
    }

    let group_start_byte = chars[group_start].0;
    let group = &text[group_start_byte..group_end_byte];

    // Only return if the group contains at least one internal dot
    if group.contains('.') {
        group
    } else {
        // Single token without dots — return it for abbreviation matching
        group
    }
}

fn is_opener(ch: char) -> bool {
    matches!(ch, '(' | '[' | '{' | '"' | '\'' | '\u{201C}' | '\u{2018}')
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AbbrevKind {
    Strong,
    Internal,
    Weak,
}

fn classify_abbrev(normalized: &str) -> Option<AbbrevKind> {
    if STRONG_ABBREVS.contains(&normalized) {
        return Some(AbbrevKind::Strong);
    }
    if INTERNAL_ABBREVS.contains(&normalized) {
        return Some(AbbrevKind::Internal);
    }
    if WEAK_ABBREVS.contains(&normalized) {
        return Some(AbbrevKind::Weak);
    }
    None
}

fn is_terminal(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '\u{2026}')
}

fn is_closer(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\'' | ')' | ']' | '}' | '\u{201D}' | '\u{2019}' | '\u{00BB}'
    )
}

fn is_decimal_period(chars: &[(usize, char)], i: usize) -> bool {
    if i == 0 || i + 1 >= chars.len() {
        return false;
    }
    chars[i - 1].1.is_ascii_digit() && chars[i + 1].1.is_ascii_digit()
}

fn next_starts_uppercase(chars: &[(usize, char)], start: usize) -> bool {
    for &(_, ch) in chars.get(start..).unwrap_or(&[]) {
        if ch.is_whitespace() {
            continue;
        }
        return ch.is_uppercase();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_split() {
        let sents = split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(sents.len(), 3);
        assert_eq!(sents[0], "Hello world.");
        assert_eq!(sents[1], "How are you?");
        assert_eq!(sents[2], "I am fine!");
    }

    #[test]
    fn title_mr() {
        let sents = split_sentences("Mr. Prosser was angry. Arthur blinked.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].starts_with("Mr. Prosser"));
    }

    #[test]
    fn title_dr() {
        let sents = split_sentences("Dr. Smith arrived. He left.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].contains("Dr. Smith"));
    }

    #[test]
    fn title_at_eof() {
        let sents = split_sentences("He met Dr.");
        assert_eq!(sents.len(), 1);
    }

    #[test]
    fn decimal_number() {
        let sents = split_sentences("It cost 42.50 dollars. That was absurd.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].contains("42.50"));
    }

    #[test]
    fn ellipsis_no_split() {
        let sents = split_sentences("Wait... what happened next?");
        assert_eq!(sents.len(), 1, "got: {:?}", sents);
    }

    #[test]
    fn ellipsis_before_uppercase() {
        let sents = split_sentences("He paused... Then spoke.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn weak_etc_before_uppercase() {
        let sents = split_sentences("He bought tea, biscuits, etc. Then left.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn weak_etc_before_lowercase() {
        let sents = split_sentences("He bought tea, biscuits, etc. and went home.");
        assert_eq!(sents.len(), 1, "got: {:?}", sents);
    }

    #[test]
    fn internal_eg() {
        let sents = split_sentences("For example, e.g. dolphins, mice, and doors. It worked.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn initialism_hg() {
        let sents = split_sentences("H.G. Wells was mentioned. Adams continued.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].contains("H.G. Wells"));
    }

    #[test]
    fn initialism_bbc() {
        let sents = split_sentences("The B.B.C. broadcast it. Nobody listened.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].contains("B.B.C."));
    }

    #[test]
    fn dialog_exclamation_lowercase() {
        // ! followed by lowercase dialog tag → no split at !
        let sents = split_sentences(r#""Go!" he said. She went."#);
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn dialog_exclamation_uppercase() {
        // ! followed by uppercase → split
        let sents = split_sentences(r#""Go!" She went."#);
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn number_abbreviation_no() {
        let sents = split_sentences("No. 42 was missing. Then it returned.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
        assert!(sents[0].contains("No. 42"));
    }

    #[test]
    fn pm_time_split() {
        let sents = split_sentences("He arrived at 3 p.m. Then he left.");
        assert_eq!(sents.len(), 2, "got: {:?}", sents);
    }

    #[test]
    fn pm_time_eof() {
        let sents = split_sentences("He arrived at 3 p.m.");
        assert_eq!(sents.len(), 1, "got: {:?}", sents);
    }

    #[test]
    fn empty_text() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn single_sentence_no_terminal() {
        let sents = split_sentences("Hello world");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], "Hello world");
    }

    #[test]
    fn adams_passage() {
        let text = "The ships hung in the sky in much the same way that bricks don't. \
                    Mr. Prosser was not a man who would usually command attention. \
                    He had a remarkable talent for being unremarkable.";
        let sents = split_sentences(text);
        assert_eq!(sents.len(), 3, "got: {:?}", sents);
    }
}
