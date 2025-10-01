use std::path::PathBuf;

use clap::{Parser, Subcommand};
use forge::{document::{DocumentProcessor}, llm::LLMEngine, rag::{EntityExtractor, ForgeIndex, QueryEngine}};
use forge::graph::KnowledgeGraph;
use forge::vector::{InMemoryVectorStore, VectorStore};

#[derive(Parser, Debug)]
#[command(name = "forge", about = "Forge: Offline GraphRAG in Rust (scaffold)")]
struct Cli {
    /// Optional path to a local model file (e.g., GGUF)
    #[arg(long, global = true)]
    model_path: Option<PathBuf>,

    /// Optional path to tokenizer.json
    #[arg(long, global = true)]
    tokenizer_path: Option<PathBuf>,

    /// Device selection (cpu, cuda)
    #[arg(long, global = true, default_value = "cpu")]
    device: String,

    /// Optional path to a config file (TOML)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download/setup the recommended model to models/
    SetupModel { #[arg(long)] url: Option<String>, #[arg(long)] out: Option<PathBuf> },
    /// Test LLM generation using Candle backend
    LlmTest { prompt: String, #[arg(long)] tokenizer_path: Option<PathBuf>, #[arg(long)] max_tokens: Option<usize>, #[arg(long)] temperature: Option<f64>, #[arg(long)] top_p: Option<f64>, #[arg(long)] top_k: Option<usize> },
    /// Index documents in a directory and build a knowledge graph
    Index { input: PathBuf, output: PathBuf },
    /// Query an existing index
    Query { query: String, index: PathBuf },
    /// Export graph
    Export { index: PathBuf, format: String, output: PathBuf },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Merge config values if provided (no mutation of cli)
    let mut model_path = cli.model_path.clone();
    let mut device = cli.device.clone();
    let mut tokenizer_path = cli.tokenizer_path.clone();
    if let Some(cfg_path) = &cli.config {
        if let Ok(cfg) = forge::config::Config::load(cfg_path) {
            if model_path.is_none() { model_path = cfg.model_path; }
            if device == "cpu" { if let Some(d) = cfg.device { device = d; } }
            if tokenizer_path.is_none() { tokenizer_path = cfg.tokenizer_json; }
        }
    }

    match cli.command {
        Commands::SetupModel { url, out } => {
            let default_url = "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q3_K_L.gguf?download=true".to_string();
            let url = url.unwrap_or(default_url);
            let out = out.unwrap_or(PathBuf::from("models/Qwen3-0.6B-Q3_K_L.gguf"));
            let path = forge::llm::downloader::ensure_model(&out, &url).await?;
            println!("Model downloaded to {}", path.display());
        }
        Commands::LlmTest { prompt, tokenizer_path: tp_cli, max_tokens, temperature, top_p, top_k } => {
            let model_path = model_path.clone().unwrap_or_else(|| PathBuf::from("models/Qwen3-0.6B-Q3_K_L.gguf"));
            let tokenizer_effective = tp_cli.or(tokenizer_path.clone());
            let engine = LLMEngine::with_candle(model_path, Some(device.clone()), tokenizer_effective, max_tokens, temperature, top_p, top_k);
            let out = engine.generate(&prompt).await?;
            println!("{}", out);
        }
        Commands::Index { input, output } => {
            index_cmd_with_cfg(&input, &output, &model_path, &device).await?;
        }
        Commands::Query { query, index } => {
            query_cmd_with_cfg(&query, &index, &cli.model_path, &cli.device).await?;
        }
        Commands::Export { index, format, output } => {
            export_cmd(&index, &format, &output).await?;
        }
    }

    Ok(())
}

async fn index_cmd_with_cfg(input: &PathBuf, output: &PathBuf, model_path: &Option<PathBuf>, device: &str) -> anyhow::Result<()> {
    let mut graph = KnowledgeGraph::default();
    let mut chunks_all = Vec::new();
    let llm = match model_path {
        Some(p) => LLMEngine::with_candle(p.clone(), Some(device.to_string()), None, None, None, None, None),
        None => LLMEngine::new(),
    };
    let extractor = EntityExtractor::new(llm.clone());

    // Walk directory
    for entry in walkdir::WalkDir::new(input).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            if ["txt", "text", "md", "markdown", "pdf"].contains(&ext.as_str()) {
                match DocumentProcessor::parse_path(path) {
                    Ok(chunks) => {
                        for chunk in &chunks {
                            // Extract entities
                            let (entities, relationships) = extractor.extract(chunk).await?;
                            for mut e in entities {
                                let id = graph.add_entity(e.clone());
                                // Ensure entity has id set if not
                                if graph.nodes.get(&id).is_none() {
                                    e.id = id.clone();
                                }
                            }
                            for r in relationships { graph.add_relationship(r); }
                        }
                        chunks_all.extend(chunks);
                    }
                    Err(err) => eprintln!("Failed to parse {}: {}", path.display(), err),
                }
            }
        }
    }

    let index = ForgeIndex { graph, chunks: chunks_all };
    index.save_json(output)?;
    println!("Indexed and saved to {}", output.display());
    Ok(())
}

async fn query_cmd_with_cfg(query: &str, index_path: &PathBuf, model_path: &Option<PathBuf>, device: &str) -> anyhow::Result<()> {
    let index = ForgeIndex::load_json(index_path)?;
    let llm = match model_path {
        Some(p) => LLMEngine::with_candle(p.clone(), Some(device.to_string()), None, None, None, None, None),
        None => LLMEngine::new(),
    };
    let mut vs = InMemoryVectorStore::default();

    // Insert chunk vectors
    for chunk in &index.chunks {
        let v = vs.embed_text(&chunk.text);
        vs.upsert(chunk.id.clone(), v);
    }

    let engine = QueryEngine::new(index.graph, llm, vs);
    let answer = engine.query(query).await?;
    println!("{}", answer);
    Ok(())
}

async fn export_cmd(index_path: &PathBuf, format: &str, output: &PathBuf) -> anyhow::Result<()> {
    let index = ForgeIndex::load_json(index_path)?;
    match format.to_ascii_lowercase().as_str() {
        "graphml" => {
            let xml = index.graph.to_graphml();
            std::fs::write(output, xml)?;
            println!("Exported GraphML to {}", output.display());
        }
        other => {
            anyhow::bail!("Unsupported export format: {}", other);
        }
    }
    Ok(())
}
