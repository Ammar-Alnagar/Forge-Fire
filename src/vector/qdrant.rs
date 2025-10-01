#[cfg(feature = "vector-qdrant")]
pub mod qdrant_store {
    use qdrant_client::client::QdrantClient;
    use qdrant_client::qdrant::{vectors_config::Config, Distance, PointStruct, VectorsConfig, CreateCollection, VectorParams};

    use super::super::vector::VectorStore;

    pub struct QdrantStore {
        pub client: QdrantClient,
        pub collection: String,
        pub dim: usize,
    }

    impl QdrantStore {
        pub async fn ensure_collection(&self) -> anyhow::Result<()> {
            let cfg = VectorsConfig { config: Some(Config::Params(VectorParams { size: self.dim as u64, distance: Distance::Cosine as i32, ..Default::default() })) };
            let _ = self.client.create_collection(&CreateCollection { collection_name: self.collection.clone(), vectors_config: Some(cfg), ..Default::default() }).await;
            Ok(())
        }
    }

    impl VectorStore for QdrantStore {
        fn upsert(&mut self, id: String, vector: Vec<f32>) { let _ = (id, vector); }
        fn search(&self, _query: &[f32], _k: usize) -> Vec<(String, f32)> { vec![] }
        fn embed_text(&self, text: &str) -> Vec<f32> { let mut v = vec![0f32; self.dim]; for &b in text.as_bytes() { v[(b as usize) % self.dim] += 1.0; } v }
    }
}
