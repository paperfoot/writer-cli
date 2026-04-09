//! On-disk sample representation.
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SampleSource {
    #[default]
    Unknown,
    Markdown,
    PlainText,
    TwitterArchive,
    LinkedIn,
    Obsidian,
    EmailMbox,
    Url,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub source: SampleSource,
    pub origin_path: Option<PathBuf>,
    pub context_tag: Option<String>,
    pub captured_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub content: String,
    pub metadata: SampleMetadata,
    pub content_hash: String,
}

impl Sample {
    pub fn new(content: String, metadata: SampleMetadata) -> Self {
        let content_hash = hash_content(&content);
        Self {
            content,
            metadata,
            content_hash,
        }
    }

    /// Unicode-segmenter word count, used for corpus size reports.
    pub fn word_count(&self) -> usize {
        self.content.unicode_words().count()
    }

    /// Unicode-segmenter character count, used for stylometry.
    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
}

fn hash_content(s: &str) -> String {
    // Stable, small, deterministic hash. Not cryptographic — used for
    // dedupe only.
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
