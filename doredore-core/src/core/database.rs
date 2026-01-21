use crate::core::collection::{Collection, Document};
use crate::error::Result;
use rusqlite::{params, Connection};
use std::path::Path;

pub struct Database {
    db_path: String,
}

impl Database {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref().to_string_lossy().to_string();
        let conn = Connection::open(&db_path)?;
        Self::init_schema(&conn)?;
        Ok(Self { db_path })
    }

    fn connect(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        // コレクションテーブル
        conn.execute(
            "CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // ドキュメントテーブル
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // 設定テーブル
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )",
            [],
        )?;

        // インデックス
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collections_name ON collections(name)",
            [],
        )?;

        // FTS5仮想テーブル（Full-Text Search）
        // キーワード検索用の転置インデックスを提供
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                document_id UNINDEXED,  -- ドキュメントIDは検索対象外（JOINキーとして使用）
                content,                -- 検索対象のテキストカラム
                tokenize = 'unicode61 remove_diacritics 2'  -- Unicode対応トークナイザー
            )",
            // tokenize設定:
            // - unicode61: Unicode 6.1の単語境界ルールを使用
            // - remove_diacritics 2: アクセント記号を除去してマッチング精度を向上
            // 注意: CJK言語（日本語・中国語・韓国語）の分割は不完全
            [],
        )?;

        Ok(())
    }

    // コレクション管理

    pub fn create_collection(&self, name: &str, description: Option<&str>) -> Result<i64> {
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO collections (name, description) VALUES (?1, ?2)",
            params![name, description],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_collection(&self, name: &str) -> Result<Collection> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name, c.description,
                    COUNT(d.id) as document_count,
                    c.created_at, c.updated_at
             FROM collections c
             LEFT JOIN documents d ON c.id = d.collection_id
             WHERE c.name = ?1
             GROUP BY c.id",
        )?;

        let collection = match stmt.query_row(params![name], |row| {
            Ok(Collection::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
            ))
        }) {
            Ok(collection) => collection,
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                return Err(crate::error::Error::CollectionNotFound(name.to_string()))
            }
            Err(err) => return Err(err.into()),
        };

        Ok(collection)
    }

    pub fn get_collection_by_id(&self, id: i64) -> Result<Collection> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name, c.description,
                    COUNT(d.id) as document_count,
                    c.created_at, c.updated_at
             FROM collections c
             LEFT JOIN documents d ON c.id = d.collection_id
             WHERE c.id = ?1
             GROUP BY c.id",
        )?;

        let collection = stmt.query_row(params![id], |row| {
            Ok(Collection::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
            ))
        })?;

        Ok(collection)
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name, c.description,
                    COUNT(d.id) as document_count,
                    c.created_at, c.updated_at
             FROM collections c
             LEFT JOIN documents d ON c.id = d.collection_id
             GROUP BY c.id
             ORDER BY c.created_at DESC",
        )?;

        let collections = stmt
            .query_map([], |row| {
                Ok(Collection::new(
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(collections)
    }

    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        let conn = self.connect()?;
        let rows_affected = conn.execute("DELETE FROM collections WHERE name = ?1", params![name])?;
        Ok(rows_affected > 0)
    }

    // ドキュメント管理

    pub fn add_document(
        &self,
        collection_id: i64,
        content: &str,
        embedding: &[f32],
        metadata: Option<&serde_json::Value>,
    ) -> Result<i64> {
        let conn = self.connect()?;
        let embedding_bytes = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        let metadata_json = metadata.map(|m| serde_json::to_string(m)).transpose()?;

        conn.execute(
            "INSERT INTO documents (collection_id, content, embedding, metadata)
             VALUES (?1, ?2, ?3, ?4)",
            params![collection_id, content, embedding_bytes, metadata_json],
        )?;

        let document_id = conn.last_insert_rowid();

        // FTSテーブルにも挿入（キーワード検索用のインデックスを構築）
        // documentsテーブルとdocuments_ftsテーブルの同期を保つ
        conn.execute(
            "INSERT INTO documents_fts (document_id, content) VALUES (?1, ?2)",
            params![document_id, content],
        )?;

        Ok(document_id)
    }

    pub fn get_document(&self, document_id: i64) -> Result<Document> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT d.id, d.collection_id, c.name, d.content, d.metadata,
                    d.created_at, d.updated_at
             FROM documents d
             JOIN collections c ON d.collection_id = c.id
             WHERE d.id = ?1",
        )?;

        let document = stmt.query_row(params![document_id], |row| {
            let metadata_str: Option<String> = row.get(4)?;
            let metadata = metadata_str
                .map(|s| serde_json::from_str(&s))
                .transpose()
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

            Ok(Document::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                metadata,
                row.get(5)?,
                row.get(6)?,
            ))
        })?;

        Ok(document)
    }

    pub fn list_documents(
        &self,
        collection_id: Option<i64>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Document>> {
        let conn = self.connect()?;
        let query = if let Some(cid) = collection_id {
            format!(
                "SELECT d.id, d.collection_id, c.name, d.content, d.metadata,
                        d.created_at, d.updated_at
                 FROM documents d
                 JOIN collections c ON d.collection_id = c.id
                 WHERE d.collection_id = {}
                 ORDER BY d.created_at DESC
                 LIMIT {} OFFSET {}",
                cid, limit, offset
            )
        } else {
            format!(
                "SELECT d.id, d.collection_id, c.name, d.content, d.metadata,
                        d.created_at, d.updated_at
                 FROM documents d
                 JOIN collections c ON d.collection_id = c.id
                 ORDER BY d.created_at DESC
                 LIMIT {} OFFSET {}",
                limit, offset
            )
        };

        let mut stmt = conn.prepare(&query)?;

        let documents = stmt
            .query_map([], |row| {
                let metadata_str: Option<String> = row.get(4)?;
                let metadata = metadata_str
                    .map(|s| serde_json::from_str(&s))
                    .transpose()
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                Ok(Document::new(
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    metadata,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(documents)
    }

    pub fn update_document(
        &self,
        document_id: i64,
        content: Option<&str>,
        embedding: Option<&[f32]>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<bool> {
        if content.is_none() && embedding.is_none() && metadata.is_none() {
            return Ok(false);
        }

        let conn = self.connect()?;
        let mut updates = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(c) = content {
            updates.push("content = ?");
            params_vec.push(Box::new(c.to_string()));
        }

        if let Some(e) = embedding {
            updates.push("embedding = ?");
            let embedding_bytes = e.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            params_vec.push(Box::new(embedding_bytes));
        }

        if let Some(m) = metadata {
            updates.push("metadata = ?");
            let metadata_json = serde_json::to_string(m)?;
            params_vec.push(Box::new(metadata_json));
        }

        updates.push("updated_at = CURRENT_TIMESTAMP");

        let query = format!("UPDATE documents SET {} WHERE id = ?", updates.join(", "));

        params_vec.push(Box::new(document_id));

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();

        let rows_affected = conn.execute(&query, params_refs.as_slice())?;

        Ok(rows_affected > 0)
    }

    pub fn delete_document(&self, document_id: i64) -> Result<bool> {
        let conn = self.connect()?;
        let rows_affected = conn.execute("DELETE FROM documents WHERE id = ?1", params![document_id])?;
        Ok(rows_affected > 0)
    }

    pub fn get_all_documents_with_embeddings(
        &self,
        collection_ids: Option<&[i64]>,
    ) -> Result<Vec<(i64, String, Vec<f32>, String)>> {
        let conn = self.connect()?;
        let query = if let Some(cids) = collection_ids {
            let placeholders = cids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            format!(
                "SELECT d.id, d.content, d.embedding, c.name
                 FROM documents d
                 JOIN collections c ON d.collection_id = c.id
                 WHERE d.collection_id IN ({})",
                placeholders
            )
        } else {
            "SELECT d.id, d.content, d.embedding, c.name
             FROM documents d
             JOIN collections c ON d.collection_id = c.id"
                .to_string()
        };

        let mut stmt = conn.prepare(&query)?;

        let row_mapper = |row: &rusqlite::Row| -> rusqlite::Result<(i64, String, Vec<f32>, String)> {
            let id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let embedding_bytes: Vec<u8> = row.get(2)?;
            let collection_name: String = row.get(3)?;

            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            Ok((id, content, embedding, collection_name))
        };

        let results = if let Some(cids) = collection_ids {
            let params_refs: Vec<&dyn rusqlite::ToSql> =
                cids.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
            stmt.query_map(params_refs.as_slice(), row_mapper)?
        } else {
            stmt.query_map([], row_mapper)?
        };

        Ok(results.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn document_exists_by_metadata(
        &self,
        collection_id: i64,
        key: &str,
        value: &str,
    ) -> Result<bool> {
        let conn = self.connect()?;
        let json_path = format!("$.{}", key);
        let mut stmt = conn.prepare(
            "SELECT 1 FROM documents
             WHERE collection_id = ?1
               AND CAST(json_extract(metadata, ?2) AS TEXT) = ?3
             LIMIT 1",
        )?;
        let mut rows = stmt.query(params![collection_id, json_path, value])?;
        Ok(rows.next()?.is_some())
    }

    /// キーワード検索（FTS5 + LIKE検索の2段階フォールバック）
    ///
    /// 英語と日本語の両方に対応した柔軟なキーワード検索を実装
    ///
    /// # 検索戦略
    /// 1. **第1段階: FTS5検索（高速・高精度）**
    ///    - SQLiteのFull-Text Search 5を使用
    ///    - BM25アルゴリズムでランキング
    ///    - 英語の単語分割に最適化
    ///    - 速度: O(log n)（インデックス使用）
    ///
    /// 2. **第2段階: LIKE検索（フォールバック）**
    ///    - FTS5で結果がない場合に自動的に実行
    ///    - 日本語やCJK言語に対応
    ///    - パターンマッチング: `%キーワード%`
    ///    - 速度: O(n)（全件スキャン）
    ///
    /// # 引数
    /// * `query` - 検索キーワード
    /// * `collection_ids` - 検索対象のコレクションID（Noneの場合は全コレクション）
    ///
    /// # 戻り値
    /// `Vec<(document_id, content, score, collection_name)>`
    /// * FTS5の場合: スコアはBM25スコア（負の値、小さいほど良い）
    /// * LIKE検索の場合: スコアは固定値1.0
    pub fn keyword_search(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
    ) -> Result<Vec<(i64, String, f32, String)>> {
        // まずFTS5で検索を試みる（英語などに最適）
        let fts_results = self.keyword_search_fts5(query, collection_ids);

        // FTS5が成功して結果があればそれを返す
        if let Ok(results) = &fts_results {
            if !results.is_empty() {
                return Ok(results.clone());
            }
        }

        // FTS5が失敗または結果が空の場合、LIKE検索にフォールバック
        // 日本語やCJK言語でも確実にマッチングできる
        self.keyword_search_like(query, collection_ids)
    }

    /// FTS5による全文検索
    ///
    /// SQLiteのFull-Text Search 5とBM25アルゴリズムを使用した高速検索
    ///
    /// # FTS5の特徴
    /// - **BM25ランキング**: TF-IDF改良版のランキングアルゴリズム
    /// - **トークナイザー**: unicode61（英語など欧米言語に最適化）
    /// - **インデックス**: 転置インデックスで高速検索
    /// - **制限**: 日本語などCJK言語は単語分割が不完全
    ///
    /// # BM25スコア
    /// - 負の値を返す（SQLiteのbm25()関数の仕様）
    /// - スコアが小さいほど関連性が高い
    /// - 後で正規化が必要（enricher.rsで実施）
    ///
    /// # 引数
    /// * `query` - 検索クエリ（FTS5クエリ構文）
    /// * `collection_ids` - 検索対象のコレクションID
    fn keyword_search_fts5(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
    ) -> Result<Vec<(i64, String, f32, String)>> {
        let conn = self.connect()?;
        // SQLクエリを構築
        // MATCH演算子: FTS5の全文検索を実行
        // bm25(documents_fts): BM25スコアを計算（負の値）
        let query_sql = if let Some(cids) = collection_ids {
            // 特定のコレクションに絞り込む場合
            let placeholders = cids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            format!(
                "SELECT fts.document_id, d.content, bm25(documents_fts) as score, c.name
                 FROM documents_fts fts
                 JOIN documents d ON fts.document_id = d.id
                 JOIN collections c ON d.collection_id = c.id
                 WHERE documents_fts MATCH ?1 AND d.collection_id IN ({})
                 ORDER BY score",  // BM25スコアの昇順（小さい = 高関連）
                placeholders
            )
        } else {
            // 全コレクションを対象にする場合
            "SELECT fts.document_id, d.content, bm25(documents_fts) as score, c.name
             FROM documents_fts fts
             JOIN documents d ON fts.document_id = d.id
             JOIN collections c ON d.collection_id = c.id
             WHERE documents_fts MATCH ?1
             ORDER BY score"
                .to_string()
        };

        let mut stmt = conn.prepare(&query_sql)?;

        let row_mapper = |row: &rusqlite::Row| -> rusqlite::Result<(i64, String, f32, String)> {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        };

        let results = if let Some(cids) = collection_ids {
            let mut params: Vec<&dyn rusqlite::ToSql> = vec![&query];
            let cid_params: Vec<&dyn rusqlite::ToSql> =
                cids.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
            params.extend(cid_params);
            stmt.query_map(params.as_slice(), row_mapper)?
        } else {
            stmt.query_map([query], row_mapper)?
        };

        Ok(results.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// LIKE検索による検索（日本語・CJK言語対応）
    ///
    /// SQLのLIKE演算子を使った単純なパターンマッチング
    /// FTS5が対応していない日本語などのCJK言語でも確実に動作する
    ///
    /// # 引数
    /// * `query` - 検索クエリ
    /// * `collection_ids` - 検索対象のコレクションID
    fn keyword_search_like(
        &self,
        query: &str,
        collection_ids: Option<&[i64]>,
    ) -> Result<Vec<(i64, String, f32, String)>> {
        let conn = self.connect()?;
        let pattern = format!("%{}%", query);
        let query_sql = if let Some(cids) = collection_ids {
            let placeholders = cids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            format!(
                "SELECT d.id, d.content, 1.0 as score, c.name
                 FROM documents d
                 JOIN collections c ON d.collection_id = c.id
                 WHERE d.content LIKE ?1 AND d.collection_id IN ({})
                 ORDER BY d.created_at DESC",
                placeholders
            )
        } else {
            "SELECT d.id, d.content, 1.0 as score, c.name
             FROM documents d
             JOIN collections c ON d.collection_id = c.id
             WHERE d.content LIKE ?1
             ORDER BY d.created_at DESC"
                .to_string()
        };

        let mut stmt = conn.prepare(&query_sql)?;

        let row_mapper = |row: &rusqlite::Row| -> rusqlite::Result<(i64, String, f32, String)> {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        };

        let results = if let Some(cids) = collection_ids {
            let mut params: Vec<&dyn rusqlite::ToSql> = vec![&pattern];
            let cid_params: Vec<&dyn rusqlite::ToSql> =
                cids.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
            params.extend(cid_params);
            stmt.query_map(params.as_slice(), row_mapper)?
        } else {
            stmt.query_map([pattern], row_mapper)?
        };

        Ok(results.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}
