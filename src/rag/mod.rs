use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{document::Chunk, graph::{Entity, KnowledgeGraph, Relationship}, llm::LLMEngine, Result};
use crate::vector::VectorStore;

#[derive(Debug, Clone)]
pub struct EntityExtractor {
    pub llm: LLMEngine,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtractedRelationship {
    pub source: String,
    pub target: String,
    pub rel_type: String,
    pub description: Option<String>,
    pub strength: Option<f32>,
}

impl EntityExtractor {
    pub fn new(llm: LLMEngine) -> Self { Self { llm } }

    pub async fn extract(&self, chunk: &Chunk) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        // Try LLM-powered extraction; fallback to heuristic if LLM not active or parsing fails.
        let prompt = format!(
            "You are an entity extraction system. Extract entities and relationships.\n\
            Return strict JSON with fields: entities, relationships.\n\
            entities: [{{name, entity_type, description}}]\n\
            relationships: [{{source, target, rel_type, description, strength}}]\n\
            Text: \n{}",
            chunk.text
        );
        if let Ok(text) = self.llm.generate(&prompt).await {
            if let Some((ents, rels)) = parse_extraction_json(&text) {
                let entities: Vec<Entity> = ents.into_iter().map(|e| Entity{
                    id: String::new(),
                    name: e.name,
                    entity_type: e.entity_type,
                    description: e.description.unwrap_or_default(),
                    source_chunks: vec![chunk.id.clone()],
                }).collect();
                let relationships: Vec<Relationship> = rels.into_iter().map(|r| Relationship{
                    source: r.source,
                    target: r.target,
                    rel_type: r.rel_type,
                    description: r.description.unwrap_or_default(),
                    strength: r.strength.unwrap_or(1.0),
                }).collect();
                return Ok((entities, relationships));
            }
        }
        // Heuristic fallback
        let mut names = collect_capitalized_terms(&chunk.text);
        names.truncate(16);
        let entities: Vec<Entity> = names.iter().map(|name| Entity {
            id: String::new(),
            name: name.clone(),
            entity_type: "Concept".to_string(),
            description: String::new(),
            source_chunks: vec![chunk.id.clone()],
        }).collect();
        let relationships: Vec<Relationship> = Vec::new();
        Ok((entities, relationships))
    }
}

fn parse_extraction_json(text: &str) -> Option<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
    // Find first JSON block in response and parse
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let json = &text[start..=end];
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let ents: Vec<ExtractedEntity> = serde_json::from_value(v.get("entities")?.clone()).ok()?;
    let rels: Vec<ExtractedRelationship> = serde_json::from_value(v.get("relationships")?.clone()).ok()?;
    Some((ents, rels))
}

fn collect_capitalized_terms(text: &str) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut set = BTreeSet::new();
    for token in text.split(|c: char| !c.is_alphanumeric() && c != '-' ) {
        if token.len() > 1 && token.chars().next().unwrap().is_uppercase() {
            set.insert(token.to_string());
        }
    }
    set.into_iter().collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeIndex {
    pub graph: KnowledgeGraph,
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Clone, Copy)]
pub enum CommunityAlg { LabelPropagation }

pub struct CommunityDetector;

impl CommunityDetector {
    pub fn detect(&self, graph: &KnowledgeGraph) -> Vec<Vec<String>> {
        // Simple label propagation over entity-id space.
        use std::collections::HashMap;
        let mut label: HashMap<&str, String> = graph.nodes.keys().map(|id| (id.as_str(), id.clone())).collect();
        let mut changed = true;
        let edges = &graph.edges;
        let neighbors = |id: &str| -> Vec<&str> {
            let mut v = Vec::new();
            for e in edges {
                if e.source == id { v.push(e.target.as_str()); }
                else if e.target == id { v.push(e.source.as_str()); }
            }
            v
        };
        let mut iters = 0;
        while changed && iters < 20 {
            changed = false;
            iters += 1;
            for id in graph.nodes.keys() {
                let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                for n in neighbors(id) {
                    let l = label.get(n).map(|s| s.as_str()).unwrap_or(n);
                    *counts.entry(l).or_default() += 1;
                }
                if let Some((best, _)) = counts.into_iter().max_by_key(|(_, c)| *c) {
                    if best != label.get(id.as_str()).map(|s| s.as_str()).unwrap_or(id) {
                        label.insert(id, best.to_string());
                        changed = true;
                    }
                }
            }
        }
        let mut groups: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for (id, l) in label.into_iter() { groups.entry(l).or_default().push(id.to_string()); }
        groups.into_values().collect()
    }
}

pub struct SummaryGenerator { pub llm: LLMEngine }

impl SummaryGenerator {
    pub fn new(llm: LLMEngine) -> Self { Self { llm } }
    pub async fn generate(&self, community: &[String], graph: &KnowledgeGraph) -> Result<String> {
        let names: Vec<String> = community.iter().filter_map(|id| graph.nodes.get(id).map(|e| e.name.clone())).collect();
        let prompt = format!("Summarize the theme connecting these entities: {}", names.join(", "));
        self.llm.generate(&prompt).await
    }
}

impl ForgeIndex {
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let data = fs::read_to_string(path)?;
        let idx: ForgeIndex = serde_json::from_str(&data)?;
        Ok(idx)
    }
}

#[derive(Debug, Clone)]
pub struct QueryEngine<VS: VectorStore> {
    pub graph: KnowledgeGraph,
    pub llm: LLMEngine,
    pub vector_store: VS,
}

impl<VS: VectorStore> QueryEngine<VS> {
    pub fn new(graph: KnowledgeGraph, llm: LLMEngine, vector_store: VS) -> Self {
        Self { graph, llm, vector_store }
        }

    pub async fn query(&self, query: &str) -> Result<String> {
        // Stub: synthesize answer using available context sizes.
        let entity_count = self.graph.nodes.len();
        let edge_count = self.graph.edges.len();
        let prompt = format!(
            "Given a knowledge graph with {} entities and {} relationships, answer the user query: '{}'\nBe concise.",
            entity_count, edge_count, query
        );
        self.llm.generate(&prompt).await
    }
}
