
# ANALYTICS COPILOT FOR SPLUNK / LOGS

I am not actually connecting or licensing to Splunk here. It is a splunk-alternative analytics system.
- it ingests logs (like Splunk)
- stores them in PostGreSQL (instead of Splunk's proprietary index)
- uses LLMs to translate natrual language into queries
- performs analytics and anomaly detection (like Splunk but AI-powered)

## Requirements
- PostGreSQL
- Python
- Pip
- Python packages listed in [requirements.txt](requirements.txt)


## Dependencies Installation instructions
```bash
# install dependencies
pip install -r requirements.txt

# install PostGreSQL
brew install postgresql@15
psql --version # check version
```

## Installing PG Vector
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
The `log_analytics` database schema is located in the [Schemas file](./config/schemas.md)
```bash
# create the database from schema
psql -d log_analytics -f ./config/schemas.md

# verify whether it worked
psql -d log_analytics -f ./config/schemas.md
```

## Create Indices
Indexes help in quickly searching across the database. The indexing strategy here works similar to a yellow pages book. 

For my use cases, it's likely that the users will search based on timestamps, log levels and of course, the actual message itself. I need the following indices:-
1. `idx_logs_timestamp` - for time-range queries (recent logs, date filters)
2. `idx_logs_level` - for filtering by severity (`ERROR`, `WARN`, etc.)
3. `idx_logs_embedding` - `HNSW` index for fast vector similarity search
Without these, your semantic search will be slow. The `HNSW` index is critical - it makes vector search logarithmic instead of linear. 

I have stored them in [indices.sql](./config/indices.sql)
```sql
psql -d log_analytics -f config/indices.sql

# verify:
psql -d log_analytics -c "\d logs"
```

### Database Schema
```sql
LLM_Splunk_Logs $ psql postgres
psql (15.15 (Homebrew))
Type "help" for help.

postgres=# \c log_analytics
You are now connected to database "log_analytics" as user "saikamat".
log_analytics=#
log_analytics=# \d logs
                                      Table "public.logs"
  Column   |           Type           | Collation | Nullable |             Default
-----------+--------------------------+-----------+----------+----------------------------------
 id        | integer                  |           | not null | nextval('logs_id_seq'::regclass)
 timestamp | timestamp with time zone |           | not null |
 level     | character varying(20)    |           |          |
 source    | character varying(255)   |           |          |
 message   | text                     |           | not null |
 metadata  | jsonb                    |           |          |
 embedding | vector(384)              |           |          |
Indexes:
    "logs_pkey" PRIMARY KEY, btree (id)
    "idx_logs_embedding" hnsw (embedding vector_cosine_ops)
    "idx_logs_level" btree (level)
    "idx_logs_timestamp" btree ("timestamp" DESC)

```



## Generate logs
```bash
python -m src.data_pipeline.etl
```
### The `log_analytics` Database
```sql
log_analytics=# SELECT id, timestamp, level, source, message, metadata FROM logs ORDER BY timestamp DESC;
 id  |           timestamp           | level |   source   |             message             |                  metadata
-----+-------------------------------+-------+------------+---------------------------------+--------------------------------------------
  83 | 2025-12-28 20:25:48.998731+01 | INFO  | nginx      | Bad gateway                     | {"status": "500"}
  38 | 2025-12-28 20:10:19.861907+01 | INFO  | postgresql | Query timeout                   | {"database": "cache"}
  78 | 2025-12-28 19:32:42.812705+01 | ERROR | systemd    | Service started                 | {"service": "postgresql"}
  12 | 2025-12-28 14:47:07.429288+01 | ERROR | auth       | Authentication failed           | {"ip": "192.168.1.104", "user": "eve"}
  18 | 2025-12-28 14:34:07.429288+01 | WARN  | systemd    | Configuration file updated      | {"service": "postgresql"}
 107 | 2025-12-28 14:20:00.95468+01  | INFO  | nginx      | Request processed               | {"status": "503"}
 116 | 2025-12-28 12:41:06.250545+01 | INFO  | nginx      | Request processed               | {"status": "200"}
  22 | 2025-12-28 10:56:13.85222+01  | INFO  | postgresql | Database connection established | {"database": "cache"}
  79 | 2025-12-28 09:13:42.812705+01 | INFO  | kernel     | Disk space running low          | {"pid": 1, "process": "systemd"}
 130 | 2025-12-28 09:03:36.827905+01 | WARN  | cron       | Schedule updated                | {"job": "cleanup"}
 125 | 2025-12-28 08:57:36.827905+01 | INFO  | nginx      | Bad gateway                     | {"status": "503"}
 119 | 2025-12-28 05:37:06.250545+01 | WARN  | networking | Packet loss detected            | {"interface": "eth0"}
  24 | 2025-12-28 03:33:13.85222+01  | INFO  | sshd       | User login failed               | {"ip": "192.168.1.102", "user": "charlie"}
  92 | 2025-12-28 02:47:54.614504+01 | INFO  | docker     | Image pulled                    | {"container": "web"}
  50 | 2025-12-28 02:13:25.761353+01 | ERROR | sshd       | User login failed               | {"ip": "192.168.1.101", "user": "bob"}
  98 | 2025-12-28 01:26:54.614504+01 | WARN  | networking | DNS resolution failed           | {"interface": "eth0"}
  48 | 2025-12-27 23:43:25.761353+01 | WARN  | cron       | Job executed                    | {"job": "backup"}
  81 | 2025-12-27 23:33:48.998731+01 | INFO  | sshd       | User login successful           | {"ip": "192.168.1.100", "user": "alice"}
  51 | 2025-12-27 20:29:31.290008+01 | INFO  | cron       | Job executed                    | {"job": "backup"}
  36 | 2025-12-27 18:21:19.861907+01 | WARN  | postgresql | Database connection established | {"database": "cache"}
  29 | 2025-12-27 16:37:13.85222+01  | INFO  | auth       | User login successful           | {"ip": "192.168.1.104", "user": "eve"}
 115 | 2025-12-27 15:15:06.250545+01 | INFO  | networking | DNS resolution failed           | {"interface": "eth1"}
  75 | 2025-12-27 14:12:42.812705+01 | WARN  | docker     | Image pulled                    | {"container": "web"}
  30 | 2025-12-27 10:50:13.85222+01  | INFO  | docker     | Container stopped               | {"container": "web"}
  23 | 2025-12-27 09:51:13.85222+01  | WARN  | sshd       | User login failed               | {"ip": "192.168.1.100", "user": "alice"}
```

## Search logs using natural questions
```bash
python -m src.data_pipeline.search
```
### Output of search results:-
```bash
(venv_llm_logs) LLM_Splunk_Logs $ python -m src.data_pipeline.search

Search results for => what were login related issues yesterday?

(50, datetime.datetime(2025, 12, 28, 2, 13, 25, 761353, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))), 'ERROR', 'sshd', 'User login failed', {'ip': '192.168.1.101', 'user': 'bob'}, 0.408474326133728)
(23, datetime.datetime(2025, 12, 27, 9, 51, 13, 852220, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))), 'WARN', 'sshd', 'User login failed', {'ip': '192.168.1.100', 'user': 'alice'}, 0.408474326133728)
(127, datetime.datetime(2025, 12, 25, 6, 4, 36, 827905, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))), 'INFO', 'sshd', 'User login failed', {'ip': '192.168.1.100', 'user': 'alice'}, 0.408474326133728)
(105, datetime.datetime(2025, 12, 26, 2, 48, 0, 954680, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))), 'INFO', 'sshd', 'User login failed', {'ip': '192.168.1.100', 'user': 'alice'}, 0.408474326133728)
(102, datetime.datetime(2025, 12, 22, 23, 56, 0, 954680, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))), 'WARN', 'sshd', 'User login failed', {'ip': '192.168.1.102', 'user': 'charlie'}, 0.408474326133728)
```



## Project Structure
- `/src` separates code from `config/docs/tests/`
- `/src/shared` gives clean reusable utilities (DB connections, embedding helpers)
- `/src/data_pipeline` for ETL code
- `/src/backend` will use APIs i guess
- `/tests` for the unit tests if any
- `data/raw` for raw log files & `/data/processed` for processed log files
- scales well when adding services

## Motivation for PostGreSQL
- Main reason -> pgvector extension
- we store vector embeddings (384-dimensional arrays) for semantic search
- PostGreSQL has a pgvector extension that:-
    - stores vectors natively as a data type: `VECTOR`
    - has efficient similarity search algorithms (HNSW, IVFFlat)
    - can index millions of vectors for fast nearest-neighbour search

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

