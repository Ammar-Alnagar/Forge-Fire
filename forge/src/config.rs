use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub model_path: Option<PathBuf>,
    pub device: Option<String>,
    pub chunk_tokens: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub tokenizer_json: Option<PathBuf>,
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&data)?;
        Ok(cfg)
    }
}
