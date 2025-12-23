CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    level VARCHAR(20),
    source VARCHAR(255),
    message TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(384)
);