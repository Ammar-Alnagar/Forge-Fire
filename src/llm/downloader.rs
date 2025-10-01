use anyhow::Context;
use std::{fs, path::{Path, PathBuf}};

pub async fn ensure_model(path: &Path, url: &str) -> anyhow::Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    // Async download using reqwest.
    let resp = reqwest::get(url).await.with_context(|| format!("GET {}", url))?;
    let bytes = resp.bytes().await.with_context(|| "reading response bytes")?;
    fs::write(path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(path.to_path_buf())
}
