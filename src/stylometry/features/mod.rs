//! Stylometric feature extractors.
//!
//! Feature selection grounded in:
//! - Writeprints (Abbasi & Chen, 2008) — lexical, syntactic, structural features
//! - LUAR (Rivera-Soto et al., EMNLP 2021) — neural authorship embeddings
//! - PAN shared tasks (2011-2025) — function words + char n-grams as top signals
//! - "Layered Insights" (EMNLP 2025) — hybrid interpretable + neural
//! - Yule (1944), Simpson (1949) — vocabulary richness measures
//! - Flesch (1948), Coleman-Liau (1975) — readability complexity signals
pub mod function_words;
pub mod lengths;
pub mod ngrams;
pub mod punctuation;
pub mod readability;
pub mod richness;
pub mod sentences;
pub mod vocabulary;
