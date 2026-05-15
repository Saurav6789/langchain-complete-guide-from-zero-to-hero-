"""
pdf_to_faiss_ollama.py
Load PDF → split → embed with OllamaEmbeddings → index with FAISS.
Works with local Ollama models (qwen2.5-coder, llama3.1) if they expose embeddings.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ===== CONFIG =====
PDF_PATH = "../data/learning_langchain.pdf"
MODEL_NAME = "qwen2.5-coder"  # or "llama3.1" — must match `ollama list`
BATCH_SIZE = 32

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[INFO] Loaded {len(docs)} source documents (pages or chunks)")

# 2) Split with metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
split_docs = splitter.split_documents(docs)
print(f"[INFO] Split into {len(split_docs)} chunks")

# 3) Prepare Ollama embeddings
emb = OllamaEmbeddings(model=MODEL_NAME)

# Quick test to ensure Ollama embeddings are available
try:
    test_vec = emb.embed_query("test")
    print(f"[INFO] Embedding test OK; dim={len(test_vec)}")
except Exception as e:
    raise RuntimeError(
        f"Ollama embeddings test failed: {e}. Check Ollama server and MODEL_NAME."
    ) from e


# 4) Batch-embed texts
def batch_embed_documents(docs_list, batch_size=BATCH_SIZE):
    texts = [d.page_content for d in docs_list]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = emb.embed_documents(batch)
        embeddings.extend(batch_embs)
    return embeddings


embs = batch_embed_documents(split_docs)
print(f"[INFO] Computed {len(embs)} embeddings")

# 5) Build FAISS index from texts + embeddings
# FAISS.from_texts will call the embedding function itself; since we already computed embeddings,
# we create Document list and pass them to FAISS.from_documents if available. We'll create Documents with metadata.
documents = []
for d, v in zip(split_docs, embs):
    # Attach embedding vector in metadata if needed by your pipeline; FAISS.from_texts usually recomputes embeddings.
    documents.append(Document(page_content=d.page_content, metadata=d.metadata))

# Use FAISS.from_texts with Ollama embeddings directly for simplicity:
index = FAISS.from_texts(
    [d.page_content for d in split_docs],
    emb,
    metadatas=[d.metadata for d in split_docs],
)

# 6) Save local index
index.save_local("faiss_index_ollama")
print("[INFO] Saved FAISS index to faiss_index_ollama/")
