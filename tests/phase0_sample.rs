use writer_cli::corpus::sample::{Sample, SampleMetadata, SampleSource};

#[test]
fn sample_computes_stable_content_hash() {
    let a = Sample::new(
        "hello world".into(),
        SampleMetadata {
            source: SampleSource::Markdown,
            origin_path: Some("/tmp/note.md".into()),
            context_tag: Some("longform".into()),
            captured_at: None,
        },
    );
    let b = Sample::new("hello world".into(), a.metadata.clone());
    assert_eq!(a.content_hash, b.content_hash);
}

#[test]
fn sample_hash_differs_for_different_content() {
    let a = Sample::new("hello world".into(), SampleMetadata::default());
    let b = Sample::new("hello wrld".into(), SampleMetadata::default());
    assert_ne!(a.content_hash, b.content_hash);
}

#[test]
fn sample_tokens_word_count_is_unicode_aware() {
    let s = Sample::new("les élèves étudient".into(), SampleMetadata::default());
    assert_eq!(s.word_count(), 3);
}
