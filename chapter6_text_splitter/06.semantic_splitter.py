"""
semantic_splitter_ollama.py
Semantic chunking using Ollama embeddings + AgglomerativeClustering.
Works with qwen2.5-coder or llama3.* via langchain_ollama.OllamaEmbeddings.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import List
import math

# ==== CONFIG ====
MODEL_NAME = "qwen2.5-coder"  # or "llama3" / "llama3.1" depending on your pull
CHUNK_UNIT_SIZE = 400  # characters per unit before embedding
CHUNK_UNIT_OVERLAP = 50
N_CLUSTERS = 10  # desired number of semantic clusters
BATCH_SIZE = 32  # embedding batch size

# ==== SAMPLE LONG DOCUMENT (replace with your text) ====
with open("../data/long_text.txt", "r", encoding="utf-8") as f:
    long_doc = f.read()

# ==== 1) Split into smaller units (sentence/paragraph-like) ====
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_UNIT_SIZE,
    chunk_overlap=CHUNK_UNIT_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

units: List[str] = splitter.split_text(long_doc)
print(
    f"[INFO] Created {len(units)} units to embed (first unit length: {len(units[0]) if units else 0})"
)

# ==== 2) Instantiate Ollama embeddings ====
# langchain_ollama provides OllamaEmbeddings which calls the local Ollama server.
emb = OllamaEmbeddings(model=MODEL_NAME)


# Helper to batch-embed texts
def embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # OllamaEmbeddings has embed_documents(...) method
        batch_vecs = emb.embed_documents(batch)
        vectors.extend(batch_vecs)
    return np.array(vectors, dtype=float)


# ==== 3) Compute embeddings ====
X = embed_texts(units)
print(f"[INFO] Embeddings shape: {X.shape}")

# ==== 4) Decide number of clusters ====
# You can choose N_CLUSTERS directly or derive it from text length.
# Example: adaptive cluster count (optional)
if N_CLUSTERS is None:
    # set clusters proportional to sqrt of units
    n_clusters = max(2, int(math.sqrt(len(units))))
else:
    n_clusters = N_CLUSTERS

print(f"[INFO] Clustering into {n_clusters} clusters")

# ==== 5) Agglomerative clustering ====
clustering = AgglomerativeClustering(n_clusters=n_clusters)
labels = clustering.fit_predict(X)
print(f"[INFO] Completed clustering. Unique labels: {np.unique(labels)}")

# ==== 6) Group units by cluster preserving original order
clusters = {}
for idx, label in enumerate(labels):
    clusters.setdefault(label, []).append((idx, units[idx]))

# Optionally sort clusters by the first occurrence to preserve reading order
ordered_clusters = sorted(clusters.items(), key=lambda kv: min(i for i, _ in kv[1]))

semantic_chunks: List[str] = []
for label, items in ordered_clusters:
    # items is list of (index, unit); sort by index
    items_sorted = sorted(items, key=lambda x: x[0])
    chunk_text = "\n".join([u for _, u in items_sorted])
    semantic_chunks.append(chunk_text)

print(f"[INFO] Generated {len(semantic_chunks)} semantic chunks")
for i, c in enumerate(semantic_chunks[:5]):
    print(f"\n--- Chunk {i} (len {len(c)} chars) ---\n{c[:500]}")

# ==== 7) (Optional) Convert to LangChain Document objects with metadata ====
from langchain_core.documents import Document

docs = []
for label, items in ordered_clusters:
    items_sorted = sorted(items, key=lambda x: x[0])
    chunk_text = "\n".join([u for _, u in items_sorted])
    metadata = {"cluster": int(label), "unit_count": len(items_sorted)}
    docs.append(Document(page_content=chunk_text, metadata=metadata))

print(f"[INFO] Created {len(docs)} Document objects for indexing")
