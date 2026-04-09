use async_trait::async_trait;
use std::path::PathBuf;
use writer_cli::backends::training::artefact::AdapterArtifact;
use writer_cli::backends::training::config::{DpoConfig, LoraConfig, TrainingProgress};
use writer_cli::backends::training::{TrainingBackend, TrainingError};
use writer_cli::backends::types::{AdapterRef, ModelId};

struct FakeTrainer;

#[async_trait]
impl TrainingBackend for FakeTrainer {
    fn name(&self) -> &str {
        "fake-trainer"
    }

    async fn train_lora(
        &self,
        config: LoraConfig,
        _on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError> {
        Ok(AdapterArtifact {
            adapter: AdapterRef::new(
                config.profile.clone(),
                PathBuf::from("/tmp/fake.safetensors"),
            ),
            base_model: config.base_model.clone(),
            steps: 100,
            final_loss: 1.23,
            training_seconds: 42,
        })
    }

    async fn train_dpo(
        &self,
        _config: DpoConfig,
        _on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError> {
        Err(TrainingError::NotImplemented)
    }
}

#[tokio::test]
async fn fake_trainer_produces_artefact() {
    let trainer = FakeTrainer;
    let cfg = LoraConfig {
        profile: "default".into(),
        base_model: "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        dataset_dir: PathBuf::from("/tmp/samples"),
        adapter_out: PathBuf::from("/tmp/adapter.safetensors"),
        rank: 16,
        alpha: 32.0,
        learning_rate: 1e-4,
        batch_size: 4,
        max_steps: 1000,
        max_seq_len: 4096,
    };

    let noop = |_p: TrainingProgress| {};
    let artefact = trainer.train_lora(cfg, &noop).await.unwrap();
    assert_eq!(artefact.steps, 100);
    assert!((artefact.final_loss - 1.23).abs() < 1e-6);
}
