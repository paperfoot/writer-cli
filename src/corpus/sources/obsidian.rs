//! Obsidian vault source parser.
use std::path::Path;

use crate::corpus::sample::{Sample, SampleSource};
use crate::corpus::sources::markdown;

use super::{Source, SourceError};

pub struct ObsidianSource;

impl Source for ObsidianSource {
    fn name(&self) -> &'static str {
        "obsidian"
    }

    fn matches(&self, path: &Path) -> bool {
        path.is_dir() && path.join(".obsidian").is_dir()
    }

    fn parse(&self, path: &Path, context: Option<&str>) -> Result<Vec<Sample>, SourceError> {
        let mut all_samples = Vec::new();

        for entry in walkdir(path)? {
            let entry_path = entry;
            if entry_path.extension().is_none_or(|e| e != "md") {
                continue;
            }

            let content = std::fs::read_to_string(&entry_path)?;
            let is_daily = detect_daily_note(&content);
            let note_context = if is_daily {
                "journal"
            } else {
                context.unwrap_or("notes")
            };

            let cleaned = strip_wikilinks(&content);
            let cleaned = strip_dataview_blocks(&cleaned);

            let mut samples =
                markdown::parse_markdown(&cleaned, &entry_path, Some(note_context));

            for sample in &mut samples {
                sample.metadata.source = SampleSource::Obsidian;
            }

            all_samples.extend(samples);
        }

        Ok(all_samples)
    }
}

fn walkdir(root: &Path) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    walk_recursive(root, &mut files)?;
    Ok(files)
}

fn walk_recursive(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), std::io::Error> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden dirs, .trash, .obsidian, templates
        if path.is_dir() {
            if name_str.starts_with('.')
                || name_str == "templates"
                || name_str == ".trash"
            {
                continue;
            }
            walk_recursive(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn detect_daily_note(content: &str) -> bool {
    // Check YAML front matter for daily/journal tags
    if let Some(rest) = content.strip_prefix("---") {
        if let Some(end) = rest.find("---") {
            let front_matter = &rest[..end].to_lowercase();
            return front_matter.contains("daily")
                || front_matter.contains("journal");
        }
    }
    false
}

pub fn strip_wikilinks(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let bytes = content.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'[' && bytes[i + 1] == b'[' {
            i += 2;
            let start = i;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if i + 1 < bytes.len() && bytes[i] == b']' && bytes[i + 1] == b']' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                i += 1;
            }
            if depth == 0 {
                let link_text = &content[start..i];
                i += 2; // skip ]]
                if let Some((_target, display)) = link_text.split_once('|') {
                    result.push_str(display);
                } else {
                    result.push_str(link_text);
                }
            } else {
                result.push_str("[[");
                result.push_str(&content[start..]);
                break;
            }
        } else {
            result.push(content[i..].chars().next().unwrap());
            i += content[i..].chars().next().unwrap().len_utf8();
        }
    }

    result
}

fn strip_dataview_blocks(content: &str) -> String {
    let mut result = String::new();
    let mut in_dataview = false;

    for line in content.lines() {
        if line.trim().starts_with("```dataview") || line.trim().starts_with("```dataviewjs") {
            in_dataview = true;
            continue;
        }
        if in_dataview && line.trim() == "```" {
            in_dataview = false;
            continue;
        }
        if !in_dataview {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}
