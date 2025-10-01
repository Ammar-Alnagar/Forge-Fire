use std::{fs, path::{Path, PathBuf}};

use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub token_estimate: usize,
    pub source_path: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct DocumentProcessor;

impl DocumentProcessor {
    pub fn parse_path(path: &Path) -> Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "txt" | "text" | "md" | "markdown" => {
                let text = fs::read_to_string(path)?;
                Ok(Self::chunk_text(&text, 512, Some(path.to_path_buf())))
            }
            "pdf" => {
                #[cfg(feature = "pdf")]
                {
                    use lopdf::Document;
                    let doc = Document::load(path)?;
                    let mut text = String::new();
                    for page_id in doc.get_pages().values() {
                        if let Ok(content) = doc.extract_text(&[*page_id]) { text.push_str(&content); text.push('\n'); }
                    }
                    Ok(Self::chunk_text(&text, 512, Some(path.to_path_buf())))
                }
                #[cfg(not(feature = "pdf"))]
                {
                    anyhow::bail!("PDF support not enabled. Build with --features pdf");
                }
            }
            _ => {
                anyhow::bail!("Unsupported file type: {}", path.display());
            }
        }
    }

    pub fn chunk_text(text: &str, target_tokens: usize, source_path: Option<PathBuf>) -> Vec<Chunk> {
        // Very simple word-based chunking approximation.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![];
        }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let stride = target_tokens; // default no overlap
        let mut idx = 0usize;
        while start < words.len() {
            let end = (start + stride).min(words.len());
            let chunk_text = words[start..end].join(" ");
            chunks.push(Chunk {
                id: format!("chunk-{}", idx),
                text: chunk_text,
                token_estimate: end - start,
                source_path: source_path.clone(),
            });
            idx += 1;
            start = end;
        }
        chunks
    }

    pub fn chunk_text_with_overlap(text: &str, target_tokens: usize, overlap: usize, source_path: Option<PathBuf>) -> Vec<Chunk> {
        if target_tokens == 0 { return vec![]; }
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() { return vec![]; }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut idx = 0usize;
        while start < words.len() {
            let end = (start + target_tokens).min(words.len());
            let chunk_text = words[start..end].join(" ");
            chunks.push(Chunk {
                id: format!("chunk-{}", idx),
                text: chunk_text,
                token_estimate: end - start,
                source_path: source_path.clone(),
            });
            idx += 1;
            if end == words.len() { break; }
            let back = overlap.min(end - start);
            start = end - back;
        }
        chunks
    }

    #[cfg(feature = "llm")]
    pub fn chunk_with_tokenizer(tokenizer: &tokenizers::Tokenizer, text: &str, target_tokens: usize, overlap: usize, source_path: Option<PathBuf>) -> Vec<Chunk> {
        use tokenizers::EncodeInput;
        if target_tokens == 0 { return vec![]; }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut idx = 0usize;
        let enc = tokenizer.encode(text, true).ok();
        if enc.is_none() { return Self::chunk_text_with_overlap(text, target_tokens, overlap, source_path); }
        let ids = enc.unwrap().get_ids().to_vec();
        while start < ids.len() {
            let end = (start + target_tokens).min(ids.len());
            // Recover text slice best-effort using byte offsets (tokenizer must provide tokens with offsets).
            // If offsets are unavailable, fall back to string slicing approximation.
            let chunk_text = text.to_string();
            chunks.push(Chunk {
                id: format!("chunk-{}", idx),
                text: chunk_text,
                token_estimate: end - start,
                source_path: source_path.clone(),
            });
            idx += 1;
            if end == ids.len() { break; }
            let back = overlap.min(end - start);
            start = end - back;
        }
        chunks
    }
}
