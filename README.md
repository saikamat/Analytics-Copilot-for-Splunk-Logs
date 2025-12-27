
# ANALYTICS COPILOT FOR SPLUNK / LOGS

I am not actually connecting or licensing to Splunk here. It is a splunk-alternative analytics system.
- it ingests logs (like Splunk)
- stores them in PostGreSQL (instead of Splunk's proprietary index)
- uses LLMs to translate natrual language into queries
- performs analytics and anomaly detection (like Splunk but AI-powered)

## Requirements
postgres


## Installation instructions
```bash
brew install postgresql@15
psql --version
```

### Installing PG Vector
```bash
psql postgres

# create database
CREATE DATABASE log_analytics;

# connect to database
\c log_analytics

# install pgvector extension
CREATE EXTENSION vector;

# verify whether it worked
SELECT * FROM pg_extension WHERE extname='vector';
```

### Issues with installing PG Vector
#### Method 1: Brew
```bash
brew install pgvector

# verify whether it worked
ls /opt/homebrew/opt/postgresql@15/share/postgresql@15/extension/vector*
```

#### Method 2: Manual Installation from source
```bash
cd \tmp

git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git

cd pgvector

## compile & install
make PG_CONFIG=/opt/homebrew/opt/postgresql@15/bin/pg_config
sudo make install PG_CONFIG=/opt/homebrew/opt/postgresql@15/bin/pg_config

# verify whether it worked
ls /opt/homebrew/opt/postgresql@15/share/postgresql@15/extension/vector*
```

## Create database
- The database schema is located in the [Schemas file](./config/schemas.md)
```bash
# create the database from schema
psql -d log_analytics -f ./config/schemas.md

# verify whether it worked
psql -d log_analytics -f ./config/schemas.md
```

## Project Structure
- `/src` separates code from `config/docs/tests/`
- `/src/shared` gives clean reusable utilities (DB connections, embedding helpers)
- `/src/data_pipeline` for ETL code
- `/src/backend` will use APIs i guess
- `/tests` for the unit tests if any
- `data/raw` for raw log files & `/data/processed` for processed log files
- scales well when adding services


## Vector Embedding
- it's a way to convert text (like a log message) into a list of numbers that captures its meaning

Example:-
- `Log message: User authentication failed for admin`
- Vector Embedding: [0.23, -0.41, 0.88, ..., 0.15] (384 numbers)
- these numbers represent meaning in math form
- Query: `authentication failure`

### Traditional Search (keyword matching)
- Finds logs containing keywords exactly `authentication` AND `failure`
- misses: `login denied`, `auth error`, `invalid credentials`


### Semantic Search (embedding similarity)
- Query: "authentication failure" → [0.25, -0.39, 0.86, ...]
Finds logs with similar embeddings:
- `User login denied for admin` → [0.24, -0.40, 0.87, ...]
- `Invalid password attempt` → [0.22, -0.38, 0.85, ...]
- `Auth token expired` → [0.26, -0.41, 0.88, ...]
When embeddings are close in vector space => similar meaning

## Motivation for PostGreSQL
- Main reason -> pgvector extension
- we store vector embeddings (384-dimensional arrays) for semantic search
- PostGreSQL has a pgvector extension that:-
    - stores vectors natively as a data type: `VECTOR`
    - has efficient similarity search algorithms (HNSW, IVFFlat)
    - can index millions of vectors for fast nearest-neighbour search

