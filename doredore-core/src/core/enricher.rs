use crate::core::{
    collection::{Collection, Document},
    database::Database,
    embedding::EmbeddingModel,
    search::{cosine_similarity, EnrichResult, SearchResult, SearchMode},
};
use crate::error::{Error, Result};
use std::path::Path;
use std::collections::HashMap;

pub struct Doredore {
    db: Database,
    embedding_model: EmbeddingModel,
}

impl Doredore {
    pub fn new<P: AsRef<Path>>(
        db_path: P,
        model: Option<&str>,
        cache_dir: Option<&str>,
    ) -> Result<Self> {
        let db = Database::new(db_path)?;
        let embedding_model = EmbeddingModel::new(model, cache_dir)?;

        Ok(Self {
            db,
            embedding_model,
        })
    }

    // コレクション管理

    pub fn create_collection(&self, name: &str, description: Option<&str>) -> Result<i64> {
        self.db.create_collection(name, description)
    }

    pub fn get_collection(&self, name: &str) -> Result<Collection> {
        self.db.get_collection(name)
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        self.db.list_collections()
    }

    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        self.db.delete_collection(name)
    }

    // ドキュメント管理

    pub fn add_document(
        &self,
        content: &str,
        collection: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<i64> {
        // コレクションIDを取得
        let coll = self.db.get_collection(collection).map_err(|_| {
            Error::CollectionNotFound(format!("Collection '{}' not found", collection))
        })?;

        // Embedding生成
        let embedding = self.embedding_model.embed(content)?;

        // ドキュメント追加
        self.db
            .add_document(coll.id, content, &embedding, metadata)
    }

    pub fn add_documents(
        &self,
        documents: Vec<String>,
        collection: &str,
        metadata: Option<Vec<serde_json::Value>>,
    ) -> Result<Vec<i64>> {
        // コレクションIDを取得
        let coll = self.db.get_collection(collection).map_err(|_| {
            Error::CollectionNotFound(format!("Collection '{}' not found", collection))
        })?;

        // Embeddingをバッチ生成
        let embeddings = self.embedding_model.embed_batch(documents.clone())?;

        // ドキュメントを追加
        let mut ids = Vec::new();
        for (i, (doc, emb)) in documents.iter().zip(embeddings.iter()).enumerate() {
            let meta = metadata.as_ref().and_then(|m| m.get(i));
            let id = self.db.add_document(coll.id, doc, emb, meta)?;
            ids.push(id);
        }

        Ok(ids)
    }

    pub fn get_document(&self, document_id: i64) -> Result<Document> {
        self.db.get_document(document_id)
    }

    pub fn list_documents(
        &self,
        collection: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Document>> {
        let collection_id = if let Some(coll_name) = collection {
            Some(self.db.get_collection(coll_name)?.id)
        } else {
            None
        };

        self.db.list_documents(collection_id, limit, offset)
    }

    pub fn document_exists_by_metadata(
        &self,
        collection: &str,
        key: &str,
        value: &str,
    ) -> Result<bool> {
        let coll = self.db.get_collection(collection).map_err(|_| {
            Error::CollectionNotFound(format!("Collection '{}' not found", collection))
        })?;
        self.db.document_exists_by_metadata(coll.id, key, value)
    }

    pub fn update_document(
        &self,
        document_id: i64,
        content: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<bool> {
        let embedding = if let Some(c) = content {
            Some(self.embedding_model.embed(c)?)
        } else {
            None
        };

        self.db.update_document(
            document_id,
            content,
            embedding.as_deref(),
            metadata,
        )
    }

    pub fn delete_document(&self, document_id: i64) -> Result<bool> {
        self.db.delete_document(document_id)
    }

    // ==================== 検索・エンリッチ ====================

    /// マルチモーダル検索のエントリーポイント
    ///
    /// 3種類の検索モード（Semantic / Keyword / Hybrid）を統一APIで提供
    ///
    /// # 引数
    /// * `query` - 検索クエリ文字列
    /// * `collection` - 検索対象の単一コレクション名
    /// * `collections` - 検索対象の複数コレクション名（collectionと排他）
    /// * `top_k` - 返す結果の最大数
    /// * `threshold` - セマンティック検索の最小スコア閾値（0.0〜1.0）
    /// * `mode` - 検索モード（Semantic / Keyword / Hybrid）
    /// * `hybrid_weights` - ハイブリッド検索の重み `(semantic_weight, keyword_weight)`
    ///
    /// # 検索モード
    /// - **Semantic**: 意味ベースの検索（埋め込みベクトル + コサイン類似度）
    /// - **Keyword**: キーワードベースの検索（FTS5 BM25 or LIKE）
    /// - **Hybrid**: 両方を組み合わせた検索（加重平均）
    ///
    /// # 戻り値
    /// スコア降順でソートされた検索結果のリスト
    pub fn search(
        &self,
        query: &str,
        collection: Option<&str>,
        collections: Option<&[String]>,
        top_k: usize,
        threshold: f32,
        mode: SearchMode,
        hybrid_weights: Option<(f32, f32)>,
    ) -> Result<Vec<SearchResult>> {
        let collection_ids = self.get_collection_ids(collection, collections)?;

        // 検索モードに応じて適切な検索関数を呼び出す
        match mode {
            SearchMode::Semantic => {
                self.semantic_search(query, collection_ids.as_deref(), top_k, threshold)
            }
            SearchMode::Keyword => {
                self.keyword_search(query, collection_ids.as_deref(), top_k)
            }
            SearchMode::Hybrid => {
                // デフォルト重み: セマンティック70% + キーワード30%
                let (semantic_weight, keyword_weight) = hybrid_weights.unwrap_or((0.7, 0.3));
                self.hybrid_search(
                    query,
                    collection_ids.as_deref(),
                    top_k,
                    threshold,
                    semantic_weight,
                    keyword_weight,
                )
            }
        }
    }

    /// セマンティック検索（意味ベース検索）
    ///
    /// Dense Embedding + Cosine Similarityを使った意味的類似性検索
    ///
    /// # アルゴリズム
    /// 1. クエリをベクトル化（BGE/E5モデル）
    /// 2. 全ドキュメントのベクトルを取得
    /// 3. コサイン類似度を計算（O(n × d)）
    /// 4. スコアでソートしてtop-kを返す
    ///
    /// # 特徴
    /// - **長所**: 言い換え・類義語に対応、多言語対応
    /// - **短所**: 計算量O(n × d)、完全一致が保証されない
    ///
    /// # スコアリング
    /// - コサイン類似度（0.0〜1.0、まれに負の値）
    /// - 1.0に近いほど意味的に類似
    ///
    /// # 引数
    /// * `query` - 検索クエリ
    /// * `collection_ids` - 対象コレクションID
    /// * `top_k` - 返す結果数
    /// * `threshold` - 最小スコア閾値
    fn semantic_search(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        // クエリのEmbeddingを生成（384次元ベクトル）
        let query_embedding = self.embedding_model.embed(query)?;

        // 全ドキュメントとEmbeddingを取得（Linear Search）
        let documents = self.db.get_all_documents_with_embeddings(collection_ids)?;

        // 各ドキュメントとの類似度を計算
        let mut results: Vec<(i64, String, f32, String)> = documents
            .into_iter()
            .map(|(id, content, embedding, coll_name)| {
                // コサイン類似度を計算
                let score = cosine_similarity(&query_embedding, &embedding);
                (id, content, score, coll_name)
            })
            // 閾値未満のドキュメントを除外
            .filter(|(_, _, score, _)| *score >= threshold)
            .collect();

        // スコアの降順でソート（高い = より類似）
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Top-K を取得してSearchResult構造体に変換
        let top_results: Vec<SearchResult> = results
            .into_iter()
            .take(top_k)
            .map(|(id, content, score, coll_name)| {
                // メタデータを取得（オプショナル）
                let doc = self.db.get_document(id).ok();
                let metadata = doc.and_then(|d| d.metadata);
                SearchResult::new(id, content, score, metadata, coll_name)
            })
            .collect();

        Ok(top_results)
    }

    /// キーワード検索（FTS5 BM25 + LIKE フォールバック）
    ///
    /// 完全一致・部分一致ベースの検索
    ///
    /// # アルゴリズム
    /// 1. FTS5でBM25検索を試行（英語に最適）
    /// 2. 結果がなければLIKE検索にフォールバック（日本語対応）
    ///
    /// # 特徴
    /// - **長所**: 正確なキーワードマッチング、高速（FTS5使用時）
    /// - **短所**: 言い換えや類義語に対応できない
    ///
    /// # スコアリング
    /// - FTS5: BM25スコア → Sigmoid正規化（0〜1）
    /// - LIKE: 固定値1.0 → Sigmoid正規化（0〜1）
    ///
    /// # 引数
    /// * `query` - 検索キーワード
    /// * `collection_ids` - 対象コレクションID
    /// * `top_k` - 返す結果数
    fn keyword_search(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        // データベース層でFTS5 → LIKE のフォールバック検索を実行
        let results = self.db.keyword_search(query, collection_ids)?;

        // BM25スコアを正規化（負の値 or 固定値を0-1に）
        let top_results: Vec<SearchResult> = results
            .into_iter()
            .take(top_k)
            .map(|(id, content, bm25_score, coll_name)| {
                // BM25スコアは負の値（小さいほど良い）
                // Sigmoid関数で0-1の範囲に正規化
                // 式: σ(x) = 1 / (1 + e^(-x/10))
                // -x/10: スケーリング係数（大きな負の値を扱いやすくする）
                let normalized_score = 1.0 / (1.0 + (-bm25_score / 10.0).exp());

                // メタデータを取得
                let doc = self.db.get_document(id).ok();
                let metadata = doc.and_then(|d| d.metadata);

                SearchResult::new(id, content, normalized_score, metadata, coll_name)
            })
            .collect();

        Ok(top_results)
    }

    /// ハイブリッド検索（セマンティック + キーワード）
    ///
    /// 意味ベース検索と完全一致検索の長所を組み合わせる
    ///
    /// # アルゴリズム
    /// 1. セマンティック検索でtop_k×2件取得
    /// 2. キーワード検索でtop_k×2件取得
    /// 3. ドキュメントIDごとにスコアをマージ
    /// 4. 加重平均でハイブリッドスコアを計算
    /// 5. 再ランキングしてtop-kを返す
    ///
    /// # スコア統合式
    /// ```text
    /// hybrid_score = w_s × semantic_score + w_k × keyword_score
    /// デフォルト: 0.7 × semantic + 0.3 × keyword
    /// ```
    ///
    /// # 特徴
    /// - 意味的な理解と正確なマッチングのバランス
    /// - 片方だけに出現するドキュメントも含まれる（欠損値は0.0）
    /// - 重み調整でユースケースに最適化可能
    ///
    /// # 引数
    /// * `query` - 検索クエリ
    /// * `collection_ids` - 対象コレクションID
    /// * `top_k` - 最終的に返す結果数
    /// * `threshold` - セマンティック検索の閾値
    /// * `semantic_weight` - セマンティックスコアの重み（0.0〜1.0）
    /// * `keyword_weight` - キーワードスコアの重み（0.0〜1.0）
    fn hybrid_search(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
        top_k: usize,
        threshold: f32,
        semantic_weight: f32,
        keyword_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        // 両方の検索を実行（top_k×2で多めに取得）
        // 後でマージして再ランキングするため、候補を多めに取る
        let semantic_results = self.semantic_search(query, collection_ids, top_k * 2, threshold)?;
        let keyword_results = self.keyword_search(query, collection_ids, top_k * 2)?;

        // ドキュメントIDをキーにしたスコアマップを作成
        // 値: (content, semantic_score, keyword_score, collection_name, metadata)
        let mut score_map: HashMap<i64, (String, f32, f32, String, Option<serde_json::Value>)> =
            HashMap::new();

        // セマンティック検索の結果を追加
        for result in semantic_results {
            score_map.insert(
                result.document_id,
                (
                    result.content.clone(),
                    result.score,  // semantic_score
                    0.0,           // keyword_score（まだない）
                    result.collection_name.clone(),
                    result.metadata.clone(),
                ),
            );
        }

        // キーワード検索の結果を追加/更新
        for result in keyword_results {
            score_map
                .entry(result.document_id)
                .and_modify(|e| e.2 = result.score) // 既存エントリのkeyword_scoreを更新
                .or_insert((
                    // 新規エントリを作成（semantic_scoreは0.0）
                    result.content.clone(),
                    0.0,
                    result.score,
                    result.collection_name.clone(),
                    result.metadata.clone(),
                ));
        }

        // ハイブリッドスコアを計算
        let mut hybrid_results: Vec<(i64, String, f32, String, Option<serde_json::Value>)> =
            score_map
                .into_iter()
                .map(|(id, (content, semantic_score, keyword_score, coll_name, metadata))| {
                    // 加重平均でハイブリッドスコアを計算
                    let hybrid_score =
                        semantic_weight * semantic_score + keyword_weight * keyword_score;
                    (id, content, hybrid_score, coll_name, metadata)
                })
                .collect();

        // ハイブリッドスコアの降順でソート
        hybrid_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Top-Kを取得してSearchResult構造体に変換
        let top_results: Vec<SearchResult> = hybrid_results
            .into_iter()
            .take(top_k)
            .map(|(id, content, score, coll_name, metadata)| {
                SearchResult::new(id, content, score, metadata, coll_name)
            })
            .collect();

        Ok(top_results)
    }

    /// RAGエンリッチメント（LLMコンテキスト生成）
    ///
    /// 検索結果をLLMに渡しやすい形式に整形
    ///
    /// # 処理フロー
    /// 1. 指定されたモードで検索を実行
    /// 2. 検索結果を整形済みコンテキスト文字列に変換
    /// 3. EnrichResultとして返す
    ///
    /// # 用途
    /// LLMプロンプトに挿入するコンテキストを生成
    /// ```text
    /// [Source 1] (Score: 0.876, Collection: docs)
    /// ドキュメントの内容...
    ///
    /// [Source 2] (Score: 0.754, Collection: docs)
    /// ドキュメントの内容...
    /// ```
    ///
    /// # 引数
    /// * searchメソッドと同じパラメータ
    ///
    /// # 戻り値
    /// EnrichResult（question, context, sources）
    pub fn enrich(
        &self,
        query: &str,
        collection: Option<&str>,
        collections: Option<&[String]>,
        top_k: usize,
        threshold: f32,
        mode: SearchMode,
        hybrid_weights: Option<(f32, f32)>,
    ) -> Result<EnrichResult> {
        // 検索を実行
        let sources = self.search(
            query,
            collection,
            collections,
            top_k,
            threshold,
            mode,
            hybrid_weights,
        )?;

        // LLM向けに整形されたコンテキストを含むEnrichResultを生成
        Ok(EnrichResult::new(query.to_string(), sources))
    }

    // ヘルパーメソッド

    fn get_collection_ids(
        &self,
        collection: Option<&str>,
        collections: Option<&[String]>,
    ) -> Result<Option<Vec<i64>>> {
        if let Some(coll_name) = collection {
            let coll = self.db.get_collection(coll_name)?;
            Ok(Some(vec![coll.id]))
        } else if let Some(coll_names) = collections {
            let mut ids = Vec::new();
            for name in coll_names {
                let coll = self.db.get_collection(name)?;
                ids.push(coll.id);
            }
            Ok(Some(ids))
        } else {
            Ok(None)
        }
    }

    // CSV インポート・エクスポート

    pub fn import_csv(
        &self,
        file_path: &str,
        collection: &str,
        content_column: &str,
        metadata_columns: Option<Vec<String>>,
    ) -> Result<usize> {
        let mut reader = csv::Reader::from_path(file_path)?;
        let headers = reader.headers()?.clone();

        let content_idx = headers
            .iter()
            .position(|h| h == content_column)
            .ok_or_else(|| {
                Error::InvalidInput(format!("Content column '{}' not found", content_column))
            })?;

        let mut documents = Vec::new();
        let mut metadata_list = Vec::new();

        for result in reader.records() {
            let record = result?;

            if let Some(content) = record.get(content_idx) {
                documents.push(content.to_string());

                // メタデータを構築
                if let Some(ref meta_cols) = metadata_columns {
                    let mut meta_map = serde_json::Map::new();
                    for col_name in meta_cols {
                        if let Some(idx) = headers.iter().position(|h| h == col_name) {
                            if let Some(value) = record.get(idx) {
                                meta_map.insert(
                                    col_name.clone(),
                                    serde_json::Value::String(value.to_string()),
                                );
                            }
                        }
                    }
                    metadata_list.push(serde_json::Value::Object(meta_map));
                } else {
                    metadata_list.push(serde_json::Value::Null);
                }
            }
        }

        let count = documents.len();
        self.add_documents(documents, collection, Some(metadata_list))?;

        Ok(count)
    }

    pub fn export_csv(
        &self,
        file_path: &str,
        collection: Option<&str>,
    ) -> Result<usize> {
        let documents = self.list_documents(collection, 1000000, 0)?;

        let mut writer = csv::Writer::from_path(file_path)?;

        // ヘッダー
        writer.write_record(&["id", "collection", "content", "metadata", "created_at"])?;

        // データ
        for doc in &documents {
            let metadata_str = doc
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default())
                .unwrap_or_default();

            writer.write_record(&[
                doc.id.to_string(),
                doc.collection_name.clone(),
                doc.content.clone(),
                metadata_str,
                doc.created_at.clone(),
            ])?;
        }

        writer.flush()?;

        Ok(documents.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_doredore_initialization() {
        let temp_file = NamedTempFile::new().unwrap();
        let result = Doredore::new(temp_file.path(), Some("bge-small-en-v1.5"), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_collection_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let rag = Doredore::new(temp_file.path(), Some("bge-small-en-v1.5"), None).unwrap();

        // Create collection
        let id = rag.create_collection("test", Some("Test collection")).unwrap();
        assert!(id > 0);

        // Get collection
        let coll = rag.get_collection("test").unwrap();
        assert_eq!(coll.name, "test");

        // List collections
        let collections = rag.list_collections().unwrap();
        assert_eq!(collections.len(), 1);

        // Delete collection
        let deleted = rag.delete_collection("test").unwrap();
        assert!(deleted);
    }

    #[test]
    fn test_document_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let rag = Doredore::new(temp_file.path(), Some("bge-small-en-v1.5"), None).unwrap();

        rag.create_collection("test", None).unwrap();

        // Add document
        let id = rag.add_document("Hello, world!", "test", None).unwrap();
        assert!(id > 0);

        // Get document
        let doc = rag.get_document(id).unwrap();
        assert_eq!(doc.content, "Hello, world!");

        // List documents
        let docs = rag.list_documents(Some("test"), 10, 0).unwrap();
        assert_eq!(docs.len(), 1);

        // Delete document
        let deleted = rag.delete_document(id).unwrap();
        assert!(deleted);
    }

    #[test]
    fn test_search() {
        let temp_file = NamedTempFile::new().unwrap();
        let rag = Doredore::new(temp_file.path(), Some("bge-small-en-v1.5"), None).unwrap();

        rag.create_collection("test", None).unwrap();
        rag.add_document("永代供養とは、お墓の管理を寺院に委託する供養形態です。", "test", None)
            .unwrap();
        rag.add_document("納骨堂には、ロッカー式、仏壇式、自動搬送式などがあります。", "test", None)
            .unwrap();

        let results = rag
            .search("永代供養について", Some("test"), None, 5, 0.0, SearchMode::Semantic, None)
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_enrich() {
        let temp_file = NamedTempFile::new().unwrap();
        let rag = Doredore::new(temp_file.path(), Some("bge-small-en-v1.5"), None).unwrap();

        rag.create_collection("test", None).unwrap();
        rag.add_document("永代供養とは、お墓の管理を寺院に委託する供養形態です。", "test", None)
            .unwrap();

        let result = rag
            .enrich("永代供養について", Some("test"), None, 3, 0.0, SearchMode::Semantic, None)
            .unwrap();

        assert_eq!(result.question, "永代供養について");
        assert!(!result.context.is_empty());
        assert!(!result.sources.is_empty());
    }
}
