import os
from .etl import get_db_connection, load_embedding_model
SAMPLE_QUERY="what were login related issues yesterday?"

# loading model once at module level
embedding_model = load_embedding_model()

def search_logs(query: str, limit: int=5) -> list[dict]:
    # generate query
    conn = get_db_connection()
    cursor = conn.cursor()

    query_embedding = embedding_model.encode(query)
    query_embedding_list = query_embedding.tolist()

    sql = """
        SELECT id, timestamp, level, source, message, metadata, 
            embedding <=> %s::vector AS distance 
        FROM logs 
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cursor.execute(sql, (query_embedding_list, query_embedding_list, limit))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

if __name__ == "__main__":
   results = search_logs(SAMPLE_QUERY, 5)
   print(f"\nSearch results for => {SAMPLE_QUERY}\n")
   for row in results:
       print(row)