# Forge: Offline GraphRAG in Rust

Forge is a fully offline, on-device GraphRAG system written in Rust. It ingests documents, builds a knowledge graph of entities and relationships, and answers queries using a local LLM.

## Key Features
- Offline-first: run entirely on CPU/GPU without external API calls
- Document ingestion: PDF (feature-gated), Markdown, Text
- Graph construction: entities and relationships with export to GraphML
- Community detection (planned): Louvain/Leiden and hierarchical summaries
- Query engine: local/global/hybrid retrieval (iterative roadmap)
- CLI and Library usage

## Tech Stack
- Rust 1.75+ with Tokio
- Candle (optional) and candle-vllm (optional)
- petgraph for graph algorithms
- Optional: Qdrant client, tokenizer-based chunking, PDF parsing

## Quickstart

1. Build

```bash
cargo build
```

2. Download the recommended Qwen3 0.6B GGUF model via CLI

```bash
# Uses Hugging Face URL from lmstudio-community/Qwen3-0.6B-GGUF
cargo run -- setup-model \
  --url https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q3_K_L.gguf?download=true \
  --out models/Qwen3-0.6B-Q3_K_L.gguf
```

3. Index documents

```bash
cargo run -- index ./documents ./forge_index.json
```

4. Query with a local model path (enables vLLM backend if built with feature)

```bash
# Build with vLLM feature to enable candle-vllm backend
cargo run --features vllm -- \
  --model-path models/Qwen3-0.6B-Q3_K_L.gguf \
  --device cpu \
  query "What are the main themes?" \
  ./forge_index.json
```

5. Export graph

```bash
cargo run -- export ./forge_index.json graphml ./graph.xml
```

## Building with features

- PDF parsing: `--features pdf`
- vLLM (candle-vllm backend): `--features vllm`

Example:

```bash
cargo run --features "pdf vllm" -- \
  --model-path models/Qwen3-0.6B-Q3_K_L.gguf \
  index ./documents ./forge_index.json
```

## Candle (official) Integration

This project uses the official Candle crates when built with `--features llm`:

- Repo: https://github.com/huggingface/candle
- Crates used: `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`
- In `LLMEngine`, calling `with_candle(model_path, device, tokenizer_path, ...)` selects the Candle backend.

Quick test (scaffold):

```bash
# Download model
cargo run -- setup-model --out models/Qwen3-0.6B-Q3_K_L.gguf

# Test generation (requires --features llm and tokenizer.json)
cargo run --features llm -- \
  --model-path models/Qwen3-0.6B-Q3_K_L.gguf \
  --device cpu \
  llm-test --prompt "Hello from Forge" --tokenizer-path models/tokenizer.json --max-tokens 64
```

Notes
- Ensure you have a compatible `tokenizer.json` for Qwen3 in `models/tokenizer.json` (or pass `--tokenizer-path`).
- The included Candle backend is a scaffold and outlines the recommended steps to load GGUF, tokenize, generate, and decode.
- The default build remains functional with a stub LLM backend if `--features llm` is not enabled.

## Roadmap

- Milestone 1: Core Infrastructure (done: scaffold, CLI, local stub)
- Milestone 2: Entity Extraction (LLM-powered) – implement prompt + JSON schema parsing using local model
- Milestone 3: Graph Building – deduplication, scoring, compression, GraphML enrichment
- Milestone 4: Community Detection – Louvain/Leiden; multi-level summaries
- Milestone 5: Query Engine – retrieval strategies and LLM synthesis with citations
- Milestone 6: CLI & Polish – caching, incremental indexing, quantization

## License

Apache-2.0 or MIT (to be decided by project). Contributions welcome.
