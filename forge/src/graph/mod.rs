use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

pub type EntityId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub source_chunks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: EntityId,
    pub target: EntityId,
    pub rel_type: String,
    pub description: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub nodes: HashMap<EntityId, Entity>,
    pub edges: Vec<Relationship>,
}

impl KnowledgeGraph {
    pub fn add_entity(&mut self, mut entity: Entity) -> EntityId {
        // Deduplicate by name (case-insensitive) for now.
        if let Some((id, _)) = self.nodes.iter().find(|(_, e)| e.name.eq_ignore_ascii_case(&entity.name)) {
            return id.clone();
        }
        let base_id = sanitize_id(&entity.name);
        let mut id = base_id.clone();
        let mut i = 1;
        while self.nodes.contains_key(&id) {
            id = format!("{}-{}", base_id, i);
            i += 1;
        }
        entity.id = id.clone();
        self.nodes.insert(id.clone(), entity);
        id
    }

    pub fn add_relationship(&mut self, rel: Relationship) {
        // Avoid duplicates
        let exists = self.edges.iter().any(|r| r.source == rel.source && r.target == rel.target && r.rel_type == rel.rel_type);
        if !exists { self.edges.push(rel); }
    }

    pub fn merge_entities(&mut self, id1: &EntityId, id2: &EntityId) {
        if id1 == id2 { return; }
        if let Some(e2) = self.nodes.remove(id2) {
            if let Some(e1) = self.nodes.get_mut(id1) {
                // Merge descriptions and source chunks.
                if !e2.description.is_empty() {
                    if !e1.description.is_empty() { e1.description.push_str(" \u{2014} "); }
                    e1.description.push_str(&e2.description);
                }
                let mut set: HashSet<String> = e1.source_chunks.iter().cloned().collect();
                set.extend(e2.source_chunks.iter().cloned());
                e1.source_chunks = set.into_iter().collect();
            } else {
                // id1 missing; put e2 back
                self.nodes.insert(id2.clone(), e2);
                return;
            }
            // Rewire edges from id2 to id1
            for edge in self.edges.iter_mut() {
                if edge.source == *id2 { edge.source = id1.clone(); }
                if edge.target == *id2 { edge.target = id1.clone(); }
            }
            // Remove possible self-loops duplicates
            self.edges.retain(|r| r.source != r.target);
        }
    }

    pub fn find_entity(&self, name: &str) -> Option<&Entity> {
        self.nodes.values().find(|e| e.name.eq_ignore_ascii_case(name))
    }

    pub fn neighbors(&self, id: &EntityId) -> Vec<&Entity> {
        let mut out = Vec::new();
        for edge in &self.edges {
            if &edge.source == id {
                if let Some(e) = self.nodes.get(&edge.target) { out.push(e); }
            } else if &edge.target == id {
                if let Some(e) = self.nodes.get(&edge.source) { out.push(e); }
            }
        }
        out
    }

    pub fn to_graphml(&self) -> String {
        let mut s = String::new();
        s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        s.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
        s.push_str("  <graph id=\"G\" edgedefault=\"undirected\">\n");
        for (id, e) in &self.nodes {
            s.push_str(&format!("    <node id=\"{}\"><data key=\"label\">{}</data></node>\n", xml_escape(id), xml_escape(&e.name)));
        }
        for (i, r) in self.edges.iter().enumerate() {
            s.push_str(&format!(
                "    <edge id=\"e{}\" source=\"{}\" target=\"{}\"><data key=\"type\">{}</data></edge>\n",
                i, xml_escape(&r.source), xml_escape(&r.target), xml_escape(&r.rel_type)
            ));
        }
        s.push_str("  </graph>\n</graphml>\n");
        s
    }
}

fn sanitize_id(s: &str) -> String {
    s.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect::<String>()
}

fn xml_escape(s: &str) -> String { s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;") }
