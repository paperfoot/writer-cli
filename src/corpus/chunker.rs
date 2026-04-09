//! Split long samples into training-sized chunks.
use crate::corpus::sample::Sample;

/// Split samples that exceed max_words into smaller chunks.
/// Respects paragraph boundaries and adds 20% overlap.
pub fn chunk(samples: Vec<Sample>, max_words: usize) -> Vec<Sample> {
    let mut result = Vec::new();

    for sample in samples {
        let word_count = sample.content.split_whitespace().count();
        if word_count <= max_words {
            result.push(sample);
            continue;
        }

        let paragraphs: Vec<&str> = sample
            .content
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        let overlap_words = max_words / 5; // 20% overlap
        let mut current_chunk = String::new();
        let mut current_words = 0;
        let mut overlap_buffer: Vec<String> = Vec::new();

        for para in &paragraphs {
            let para_words = para.split_whitespace().count();

            if current_words + para_words > max_words && !current_chunk.is_empty() {
                result.push(Sample::new(
                    current_chunk.trim().to_string(),
                    sample.metadata.clone(),
                ));

                // Start new chunk with overlap from end of previous
                current_chunk = overlap_buffer.join("\n\n");
                current_words = current_chunk.split_whitespace().count();
                overlap_buffer.clear();
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para);
            current_words += para_words;

            // Track recent paragraphs for overlap
            overlap_buffer.push(para.to_string());
            let overlap_word_count: usize = overlap_buffer
                .iter()
                .map(|p| p.split_whitespace().count())
                .sum();
            if overlap_word_count > overlap_words && overlap_buffer.len() > 1 {
                overlap_buffer.remove(0);
            }
        }

        if !current_chunk.trim().is_empty() && current_chunk.split_whitespace().count() >= 5 {
            result.push(Sample::new(
                current_chunk.trim().to_string(),
                sample.metadata.clone(),
            ));
        }
    }

    result
}
