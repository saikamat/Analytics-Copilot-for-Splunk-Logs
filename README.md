
# ANALYTICS COPILOT FOR SPLUNK / LOGS

I am not actually connecting or licensing to Splunk here.

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
