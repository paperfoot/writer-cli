//! Ollama HTTP inference backend.
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use super::capabilities::{BackendCapabilities, KvQuantKind};
use super::request::GenerationRequest;
use super::response::{FinishReason, GenerationEvent, UsageStats};
use super::{BackendError, InferenceBackend, ModelListing};
use crate::backends::types::{ModelHandle, ModelId};

pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct OllamaVersion {
    version: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Option<Vec<OllamaModel>>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
    size: Option<u64>,
}

// Legacy /api/generate structs removed — we use /api/chat exclusively.

#[derive(Debug, Serialize)]
struct OllamaPullRequest {
    name: String,
    stream: bool,
}

impl OllamaBackend {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn ping(&self) -> Result<String, BackendError> {
        let resp = self
            .client
            .get(format!("{}/api/version", self.base_url))
            .send()
            .await
            .map_err(|e| {
                BackendError::Unavailable(format!(
                    "Cannot reach Ollama at {}. Is it running? Try: brew services start ollama. Error: {e}",
                    self.base_url
                ))
            })?;

        let ver: OllamaVersion = resp.json().await.map_err(|e| {
            BackendError::Backend(format!("Failed to parse Ollama version: {e}"))
        })?;

        Ok(ver.version)
    }

    fn model_id_to_ollama_tag(model: &ModelId) -> String {
        // Translate writer's owner/name format to Ollama tags.
        //
        // Strategy: find the last hyphen before a size indicator (e.g. "26b",
        // "4b", "9b") and replace it with a colon. Strip hyphens between
        // the model family name and version number.
        //
        // Examples:
        //   google/gemma-4-26b     -> gemma4:26b
        //   google/gemma-3-4b      -> gemma3:4b
        //   google/gemma-4-26b-a4b -> gemma4:26b-a4b
        //   qwen/qwen3.5-9b        -> qwen3.5:9b
        //   meta/llama3.2-3b       -> llama3.2:3b
        //   local/gemma4:26b       -> gemma4:26b  (pass-through)
        let name = model.name();

        // If name already contains a colon, it's an Ollama tag — pass through
        if name.contains(':') {
            return name.to_string();
        }

        // Find the size indicator: a segment matching \d+b or \d+b-\w+
        // Split name by hyphens and find where the size part starts
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() <= 1 {
            return name.to_string();
        }

        // Find the index of the first part that looks like a size (e.g. "26b", "4b", "9b")
        let size_idx = parts.iter().position(|p| {
            p.len() >= 2 && p.ends_with('b') && p[..p.len()-1].chars().all(|c| c.is_ascii_digit())
        });

        if let Some(idx) = size_idx {
            // Everything before the size is the model name (join without hyphens
            // to collapse version numbers: gemma-4 -> gemma4)
            let model_name: String = parts[..idx].join("");
            let size_parts: String = parts[idx..].join("-");
            format!("{model_name}:{size_parts}")
        } else {
            // No size indicator found — try last hyphen as separator
            if let Some(pos) = name.rfind('-') {
                let (base, suffix) = name.split_at(pos);
                format!("{}:{}", base.replace('-', ""), &suffix[1..])
            } else {
                name.to_string()
            }
        }
    }

    fn ollama_tag_to_model_id(tag: &str) -> ModelId {
        // Reverse mapping from Ollama tag to ModelId
        // Strip the :latest suffix if present
        let tag = tag.strip_suffix(":latest").unwrap_or(tag);

        if let Some((owner, name)) = tag.split_once('/') {
            ModelId::new(owner, name)
        } else if tag.starts_with("gemma") {
            // gemma3:26b-a4b -> google/gemma3-26b-a4b
            ModelId::new("google", tag.replace(':', "-"))
        } else if tag.starts_with("llama") {
            ModelId::new("meta", tag.replace(':', "-"))
        } else if tag.starts_with("qwen") {
            ModelId::new("qwen", tag.replace(':', "-"))
        } else {
            ModelId::new("local", tag.replace(':', "-"))
        }
    }

    async fn generate_single(
        &self,
        model_tag: &str,
        request: &GenerationRequest,
        candidate_index: u16,
    ) -> Result<Vec<GenerationEvent>, BackendError> {
        // Use /api/chat endpoint — required for Gemma 4 and newer models
        // that don't respond to /api/generate properly.
        let mut messages = Vec::new();

        if let Some(ref sys) = request.system_prompt {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": &request.prompt}));

        let body = serde_json::json!({
            "model": model_tag,
            "messages": messages,
            "stream": false,
            "options": {
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "top_k": request.params.top_k,
                "num_predict": request.params.max_tokens,
                "repeat_penalty": request.params.repetition_penalty,
            }
        });

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| BackendError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(BackendError::Backend(format!(
                "Ollama returned {status}: {text}"
            )));
        }

        let chat_resp: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BackendError::Backend(format!("Failed to parse response: {e}")))?;

        let full_text = chat_resp
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let eval_count = chat_resp
            .get("eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let prompt_eval_count = chat_resp
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let total_duration = chat_resp
            .get("total_duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let events = vec![
            GenerationEvent::Token {
                candidate_index,
                text: full_text.clone(),
                logprob: 0.0,
            },
            GenerationEvent::Done {
                candidate_index,
                finish_reason: FinishReason::Stop,
                usage: UsageStats {
                    prompt_tokens: prompt_eval_count,
                    generated_tokens: eval_count,
                    elapsed_ms: total_duration / 1_000_000,
                },
                full_text,
            },
        ];

        Ok(events)
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn name(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_lora: true,
            supports_logit_bias: false, // Not yet in our Ollama version
            supports_contrastive_decoding: false,
            supports_speculative_decoding: false,
            supports_activation_steering: false,
            kv_quant: KvQuantKind::None,
            quant_schemes: vec![],
            max_context: 8192,
            streaming: true,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|e| BackendError::Network(e.to_string()))?;

        let tags: OllamaTagsResponse = resp
            .json()
            .await
            .map_err(|e| BackendError::Backend(format!("Failed to parse model list: {e}")))?;

        Ok(tags
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| ModelListing {
                id: Self::ollama_tag_to_model_id(&m.name),
                is_downloaded: true,
                size_bytes: m.size,
            })
            .collect())
    }

    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError> {
        let tag = Self::model_id_to_ollama_tag(id);

        // Check if model exists — try multiple matching strategies because
        // the writer ModelId format (google/gemma-4-26b-a4b) and Ollama's
        // tag format (gemma3:26b-a4b) differ significantly.
        let models = self.list_models().await?;
        let tag_lower = tag.to_lowercase();
        let id_name_lower = id.name().to_lowercase();
        let exists = models.iter().any(|m| {
            m.id == *id
            || Self::model_id_to_ollama_tag(&m.id).to_lowercase() == tag_lower
            || m.id.name().to_lowercase().contains(&id_name_lower)
            || id_name_lower.contains(&m.id.name().to_lowercase())
            // Also match key parts: "26b-a4b" should match in both directions
            || {
                let parts: Vec<&str> = id.name().split('-').collect();
                parts.len() > 2 && m.id.name().contains(parts[parts.len()-2])
            }
        });

        if !exists {
            // Try to pull
            let body = OllamaPullRequest {
                name: tag.clone(),
                stream: false,
            };

            let resp = self
                .client
                .post(format!("{}/api/pull", self.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| BackendError::Network(e.to_string()))?;

            if !resp.status().is_success() {
                return Err(BackendError::ModelNotFound(id.clone()));
            }
        }

        Ok(ModelHandle(tag))
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn Stream<Item = GenerationEvent> + Send + Unpin>, BackendError> {
        let n = request.params.n_candidates as usize;
        let model_tag = &handle.0;

        if n <= 1 {
            let events = self.generate_single(model_tag, &request, 0).await?;
            return Ok(Box::new(tokio_stream::iter(events)));
        }

        // Multi-candidate: run N concurrent requests
        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..n {
            let backend = OllamaBackend::new(&self.base_url);
            let tag = model_tag.clone();
            let req = request.clone();
            tasks.spawn(async move {
                backend.generate_single(&tag, &req, i as u16).await
            });
        }

        let mut all_events = Vec::new();
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(events)) => all_events.extend(events),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(BackendError::Backend(format!("Task failed: {e}"))),
            }
        }

        Ok(Box::new(tokio_stream::iter(all_events)))
    }
}
