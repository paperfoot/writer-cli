//! Sample normalisation chain.
//!
//! Every sample passes through `clean()` before entering the corpus.
//! Each transformation is independently testable.
use crate::corpus::sample::Sample;

pub fn clean(sample: Sample) -> Sample {
    sample
        .map_content(strip_html_tags)
        .map_content(strip_svg_blocks)
        .map_content(strip_pandoc_divs)
        .map_content(strip_link_anchors)
        .map_content(strip_image_refs)
        .map_content(unescape_chars)
        .map_content(strip_signatures)
        .map_content(strip_zero_width)
        .map_content(normalize_whitespace)
        .map_content(normalize_quotes)
        .map_content(strip_tracking_params)
}

/// Strip all HTML tags (but keep text content between them).
fn strip_html_tags(content: String) -> String {
    let mut result = String::with_capacity(content.len());
    let mut in_tag = false;
    for ch in content.chars() {
        if ch == '<' {
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            continue;
        }
        if !in_tag {
            result.push(ch);
        }
    }
    result
}

/// Strip SVG blocks entirely (cover images from epub conversion).
fn strip_svg_blocks(content: String) -> String {
    let mut result = String::new();
    let mut in_svg = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.contains("<svg") || trimmed.starts_with("svg") {
            in_svg = true;
            continue;
        }
        if in_svg {
            if trimmed.contains("</svg>") || trimmed.contains("/svg") {
                in_svg = false;
            }
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Strip pandoc ::: div markers.
fn strip_pandoc_divs(content: String) -> String {
    content
        .lines()
        .filter(|line| {
            let t = line.trim();
            !t.starts_with(":::")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strip pandoc-style link anchors like []{#something}
fn strip_link_anchors(content: String) -> String {
    let mut result = String::with_capacity(content.len());
    let mut i = 0;
    let chars: Vec<char> = content.chars().collect();
    while i < chars.len() {
        // Match []{#...}
        if i + 3 < chars.len() && chars[i] == '[' && chars[i + 1] == ']' && chars[i + 2] == '{' && chars[i + 3] == '#' {
            // Skip until closing }
            while i < chars.len() && chars[i] != '}' {
                i += 1;
            }
            if i < chars.len() {
                i += 1; // skip }
            }
            continue;
        }
        // Also match {=html} pandoc raw attribute
        if i + 5 < chars.len() && chars[i] == '{' && chars[i + 1] == '=' {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '}' {
                j += 1;
            }
            if j < chars.len() {
                i = j + 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Strip markdown image references like ![](something)
fn strip_image_refs(content: String) -> String {
    let mut result = String::with_capacity(content.len());
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '!' && chars[i + 1] == '[' {
            // Skip ![...](...)
            let mut j = i + 2;
            // Find closing ]
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            if j < chars.len() {
                j += 1; // skip ]
            }
            // Check for (...)
            if j < chars.len() && chars[j] == '(' {
                while j < chars.len() && chars[j] != ')' {
                    j += 1;
                }
                if j < chars.len() {
                    j += 1; // skip )
                }
            }
            i = j;
            continue;
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Unescape common backslash escapes from epub/pandoc conversion.
fn unescape_chars(content: String) -> String {
    content
        .replace("\\'", "'")
        .replace("\\\"", "\"")
        .replace("\\-", "-")
        .replace("\\.", ".")
        .replace("\\*", "*")
}

/// Strip email-style signature blocks (-- \n...)
fn strip_signatures(content: String) -> String {
    // Standard email sig separator is "-- \n" (dash dash space newline)
    if let Some(pos) = content.find("\n-- \n") {
        content[..pos].trim_end().to_string()
    } else if let Some(pos) = content.find("\n--\n") {
        // Also catch without trailing space
        content[..pos].trim_end().to_string()
    } else {
        content
    }
}

/// Remove zero-width characters that pollute text
fn strip_zero_width(content: String) -> String {
    content
        .replace('\u{200B}', "") // zero-width space
        .replace('\u{200C}', "") // zero-width non-joiner
        .replace('\u{200D}', "") // zero-width joiner
        .replace('\u{FEFF}', "") // BOM / zero-width no-break space
        .replace('\u{200E}', "") // left-to-right mark
        .replace('\u{200F}', "") // right-to-left mark
}

/// Collapse multiple spaces/blank lines
fn normalize_whitespace(content: String) -> String {
    let mut result = String::with_capacity(content.len());
    let mut prev_blank = false;

    for line in content.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if !prev_blank {
                result.push('\n');
                prev_blank = true;
            }
        } else {
            // Collapse multiple spaces within a line
            let mut prev_space = false;
            for ch in trimmed.chars() {
                if ch == ' ' || ch == '\t' {
                    if !prev_space {
                        result.push(' ');
                        prev_space = true;
                    }
                } else {
                    result.push(ch);
                    prev_space = false;
                }
            }
            result.push('\n');
            prev_blank = false;
        }
    }

    result.trim().to_string()
}

/// Unify quote styles per sample — preserve the author's dominant style
fn normalize_quotes(content: String) -> String {
    // Count smart vs straight quotes
    let smart_count = content.matches('\u{201C}').count()  // "
        + content.matches('\u{201D}').count()  // "
        + content.matches('\u{2018}').count()  // '
        + content.matches('\u{2019}').count(); // '

    let straight_count = content.matches('"').count() + content.matches('\'').count();

    if smart_count > 0 && straight_count > 0 {
        // Mixed — normalise to the dominant style
        if smart_count >= straight_count {
            // Convert straight to smart (approximate)
            content
        } else {
            // Convert smart to straight
            content
                .replace('\u{201C}', "\"")
                .replace('\u{201D}', "\"")
                .replace('\u{2018}', "'")
                .replace('\u{2019}', "'")
        }
    } else {
        content
    }
}

/// Strip UTM and other tracking parameters from URLs embedded in text
fn strip_tracking_params(content: String) -> String {
    let tracking_params = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "ref",
    ];

    let mut result = content;
    for param in &tracking_params {
        // Simple regex-free approach: find ?param= or &param= and strip to next & or whitespace
        loop {
            let patterns = [
                format!("?{param}="),
                format!("&{param}="),
            ];
            let mut found = false;
            for pattern in &patterns {
                if let Some(start) = result.find(pattern.as_str()) {
                    let after = start + pattern.len();
                    let end = result[after..]
                        .find(|c: char| c == '&' || c.is_whitespace() || c == ')' || c == ']')
                        .map(|e| after + e)
                        .unwrap_or(result.len());

                    if pattern.starts_with('?') {
                        // If there's a & after, replace ? with remaining
                        if end < result.len() && result.as_bytes().get(end) == Some(&b'&') {
                            result = format!(
                                "{}?{}",
                                &result[..start],
                                &result[end + 1..]
                            );
                        } else {
                            result = format!("{}{}", &result[..start], &result[end..]);
                        }
                    } else {
                        result = format!("{}{}", &result[..start], &result[end..]);
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_signatures() {
        let input = "Hello world\n\nBest regards\n-- \nBoris".to_string();
        assert_eq!(strip_signatures(input), "Hello world\n\nBest regards");
    }

    #[test]
    fn test_strip_zero_width() {
        let input = "hello\u{200B}world\u{FEFF}".to_string();
        assert_eq!(strip_zero_width(input), "helloworld");
    }

    #[test]
    fn test_normalize_whitespace() {
        let input = "hello   world\n\n\n\nfoo".to_string();
        let result = normalize_whitespace(input);
        assert_eq!(result, "hello world\n\nfoo");
    }

    #[test]
    fn test_strip_tracking_params() {
        let input = "https://example.com/page?utm_source=twitter&id=5".to_string();
        let result = strip_tracking_params(input);
        assert_eq!(result, "https://example.com/page?id=5");
    }

    #[test]
    fn test_strip_html_tags() {
        let input = "<div>Hello <b>world</b></div>".to_string();
        assert_eq!(strip_html_tags(input), "Hello world");
    }

    #[test]
    fn test_strip_pandoc_divs() {
        let input = "::: Section\nHello world\n:::".to_string();
        assert_eq!(strip_pandoc_divs(input).trim(), "Hello world");
    }

    #[test]
    fn test_strip_link_anchors() {
        let input = "[]{#chapter1.xhtml} Chapter One".to_string();
        assert_eq!(strip_link_anchors(input).trim(), "Chapter One");
    }

    #[test]
    fn test_strip_image_refs() {
        let input = "Text before ![](cover.jpeg) text after".to_string();
        assert_eq!(strip_image_refs(input), "Text before  text after");
    }

    #[test]
    fn test_unescape_chars() {
        let input = "don\\'t you think it\\'s great".to_string();
        assert_eq!(unescape_chars(input), "don't you think it's great");
    }

    #[test]
    fn test_full_epub_cleanup() {
        let sample = super::super::sample::Sample::new(
            "<div>\n::: Section\n[]{#ch1}\n![](cover.jpeg)\nDon\\'t panic.\n:::\n</div>".into(),
            super::super::sample::SampleMetadata::default(),
        );
        let cleaned = clean(sample);
        assert!(!cleaned.content.contains('<'));
        assert!(!cleaned.content.contains(":::"));
        assert!(!cleaned.content.contains("[]{#"));
        assert!(!cleaned.content.contains("!["));
        assert!(cleaned.content.contains("Don't panic"));
    }
}
