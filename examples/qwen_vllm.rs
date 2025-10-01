#[cfg(feature = "llm")]
fn main() -> anyhow::Result<()> {
    use forge::llm::LLMEngine;
    use std::path::PathBuf;
    let model = PathBuf::from("models/Qwen3-0.6B-Q3_K_L.gguf");
    let engine = LLMEngine::with_vllm(model, Some("cpu".to_string()));
    let rt = tokio::runtime::Runtime::new()?;
    let out = rt.block_on(async move { engine.generate("Hello from Forge!").await })?;
    println!("{}", out);
    Ok(())
}

#[cfg(not(feature = "llm"))]
fn main() { println!("Build with --features llm to run this example."); }
