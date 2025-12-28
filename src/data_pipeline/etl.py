import os
from psycopg2 import connect
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from data.log_generator import generate_logs
import json


load_dotenv()

def get_db_connection():
    # using env variable for database connection. Python script uses this to connect to the database
    conn = connect(os.getenv("DATABASE_URL"))
    return conn

def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# loading model once at module level
embedding_model = load_embedding_model()

def insert_logs(logs: list[dict]):
    # get the database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    data_to_insert = []
    
    # embedding_model = load_embedding_model()
    for log in logs:
        message = log['message']
        
        # convert embedding into a list
        embedding = embedding_model.encode(message)
        embedding_list = embedding.tolist() # postGreSQL needs it as a list

        # convert metadata dict to JSON string
        metadata_json = json.dumps(log.get('metadata'))

        # tuple matching SQL's columns
        row = (
            log['timestamp'],
            log['level'],
            log['source'],
            log['message'],
            metadata_json,
            embedding_list
        )

        data_to_insert.append(row)

    sql = """
        INSERT INTO logs(timestamp, level, source, message, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    execute_batch(cursor, sql, data_to_insert)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Inserted {len(logs)} logs successfully.")
        


if __name__ == "__main__":
    logs = generate_logs(10)

    # insert the logs
    insert_logs(logs)