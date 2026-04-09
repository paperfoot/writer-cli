use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::request::GenerationRequest;
use writer_cli::backends::inference::response::GenerationEvent;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::types::ModelId;

#[tokio::test]
async fn ping_returns_version() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "version": "0.19.1"
        })))
        .mount(&mock_server)
        .await;

    let backend = OllamaBackend::new(&mock_server.uri());
    let version = backend.ping().await.unwrap();
    assert_eq!(version, "0.19.1");
}

#[tokio::test]
async fn list_models_parses_tags() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                {"name": "gemma3:26b-a4b", "size": 14_000_000_000_i64},
                {"name": "llama3.2:3b", "size": 2000000000}
            ]
        })))
        .mount(&mock_server)
        .await;

    let backend = OllamaBackend::new(&mock_server.uri());
    let models = backend.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert!(models[0].is_downloaded);
}

#[tokio::test]
async fn generate_produces_events() {
    let mock_server = MockServer::start().await;

    // Mock tags endpoint for load_model
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{"name": "gemma3:26b-a4b", "size": 14_000_000_000_i64}]
        })))
        .mount(&mock_server)
        .await;

    // Mock generate endpoint
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "response": "Hello, I am writing in your voice now.",
            "done": true,
            "total_duration": 1234000000,
            "eval_count": 8,
            "prompt_eval_count": 5
        })))
        .mount(&mock_server)
        .await;

    let backend = OllamaBackend::new(&mock_server.uri());
    let model: ModelId = "google/gemma-4-26b-a4b".parse().unwrap();
    let handle = backend.load_model(&model).await.unwrap();

    let req = GenerationRequest::new(model, "Write about birds".into())
        .with_n_candidates(1);

    use tokio_stream::StreamExt;
    let mut stream = backend.generate(&handle, req).await.unwrap();

    let mut got_done = false;
    let mut full_text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            GenerationEvent::Token { text, .. } => full_text = text,
            GenerationEvent::Done { .. } => got_done = true,
            _ => {}
        }
    }

    assert!(got_done, "should get Done event");
    assert!(full_text.contains("voice"), "should contain generated text: {full_text}");
}

#[tokio::test]
async fn ping_fails_with_actionable_error() {
    let backend = OllamaBackend::new("http://localhost:1");
    let result = backend.ping().await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("brew services start ollama") || err.contains("Cannot reach"),
        "error should be actionable: {err}");
}
