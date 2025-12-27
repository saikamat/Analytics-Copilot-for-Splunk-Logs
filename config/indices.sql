CREATE INDEX idx_logs_timestamp ON logs(timestamp DESC);

CREATE INDEX idx_logs_level ON logs(level);

CREATE INDEX idx_logs_embedding ON logs USING hnsw(embedding vector_cosine_ops);