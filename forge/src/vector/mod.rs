use std::collections::HashMap;

pub trait VectorStore {
    fn upsert(&mut self, id: String, vector: Vec<f32>);
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>;
    fn embed_text(&self, text: &str) -> Vec<f32>;
}

pub struct InMemoryVectorStore {
    dim: usize,
    store: HashMap<String, Vec<f32>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self { Self { dim: 256, store: HashMap::new() } }
}

impl InMemoryVectorStore {
    pub fn new(dim: usize) -> Self { Self { dim, store: HashMap::new() } }
}

impl VectorStore for InMemoryVectorStore {
    fn upsert(&mut self, id: String, vector: Vec<f32>) {
        self.store.insert(id, l2_normalize(vector));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let q = l2_normalize(query.to_vec());
        let mut scores: Vec<(String, f32)> = self
            .store
            .iter()
            .map(|(id, v)| (id.clone(), cosine_similarity(&q, v)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        // Simple 256-dim byte histogram embedding; deterministic and fast.
        let mut v = vec![0f32; self.dim];
        for &b in text.as_bytes() { v[(b as usize) % self.dim] += 1.0; }
        l2_normalize(v)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let n = a.len().min(b.len());
    for i in 0..n { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n2: f32 = v.iter().map(|x| x * x).sum();
    if n2 > 0.0 { let norm = n2.sqrt(); for x in &mut v { *x /= norm; } }
    v
}
