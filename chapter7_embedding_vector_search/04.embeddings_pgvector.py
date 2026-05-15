# embeddings_pgvector.py

import psycopg
from langchain_ollama import OllamaEmbeddings
import json

DB_URI = "postgresql://user:password@localhost:5432/vector_db"
TABLE = "documents"

# ===== CONNECT =====
conn = psycopg.connect(DB_URI)
cur = conn.cursor()

# ===== CREATE TABLE =====
cur.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1024)  -- adjust based on model
);
""")
conn.commit()

# ===== EMBEDDING MODEL =====
emb = OllamaEmbeddings(model="mxbai-embed-large")


# ===== INSERT FUNCTION =====
def insert_document(content: str, metadata: dict):
    vec = emb.embed_query(content)

    cur.execute(
        f"""
    INSERT INTO {TABLE} (content, metadata, embedding)
    VALUES (%s, %s, %s)
    """,
        (content, json.dumps(metadata), vec),
    )

    conn.commit()


# ===== SEARCH FUNCTION =====
def similarity_search(query: str, limit=3):
    qvec = emb.embed_query(query)

    cur.execute(
        f"""
    SELECT content, metadata, embedding <-> %s AS distance
    FROM {TABLE}
    ORDER BY distance
    LIMIT %s
    """,
        (qvec, limit),
    )

    return cur.fetchall()


# ===== DEMO EXECUTION =====

if __name__ == "__main__":

    print("\n[INFO] Inserting sample documents...\n")

    insert_document(
        "LangChain helps build LLM applications with chains and tools.",
        {"source": "tutorial"},
    )

    insert_document(
        "Vector databases store embeddings for similarity search.",
        {"source": "database"},
    )

    insert_document(
        "Ollama allows running LLMs locally with embeddings support.",
        {"source": "ollama"},
    )

    print("[INFO] Documents inserted successfully!")

    # ===== QUERY =====
    query = "How do embeddings help in search?"

    print(f"\n[INFO] Running similarity search for:\n'{query}'\n")

    results = similarity_search(query, limit=3)

    # ===== PRINT RESULTS =====
    for i, row in enumerate(results):
        content, metadata, distance = row
        print(f"Result {i+1}:")
        print(f"Content: {content}")
        print(f"Metadata: {metadata}")
        print(f"Distance: {distance}")
        print("-" * 50)

# ===== CLOSE CONNECTION =====
cur.close()
conn.close()
