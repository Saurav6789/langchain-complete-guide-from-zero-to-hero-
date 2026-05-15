from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

PDF_PATH = "../data/learning_langchain.pdf"  # or use TextLoader for txt files
MODEL_NAME = "mxbai-embed-large"  # or other Ollama embedding-capable model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 32

# 1) Load documents
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[INFO] Loaded {len(docs)} source docs")

# 2) Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
split_docs = splitter.split_documents(docs)
print(f"[INFO] Split into {len(split_docs)} chunks")

# 3) Create Ollama embeddings
emb = OllamaEmbeddings(model=MODEL_NAME)

# Test embedding
try:
    test_vec = emb.embed_query("hello")
    print(f"[INFO] Embedding test OK (dim={len(test_vec)})")
except Exception as e:
    raise RuntimeError(
        "Ollama embedding failed; check Ollama server and model name"
    ) from e

# 4) Build FAISS index directly (FAISS.from_texts will call emb internally)
texts = [d.page_content for d in split_docs]
metadatas = [d.metadata for d in split_docs]
index = FAISS.from_texts(texts, emb, metadatas=metadatas)

# 5) Save index
index.save_local("faiss_index_ollama")
print("[INFO] Saved FAISS index to 'faiss_index_ollama'")

# 6) Search example
query = "What are the key findings about LLM-Based Embeddings?"
results = index.similarity_search(query, k=5)
for i, r in enumerate(results):
    print(f"\nResult {i+1} - source metadata: {r.metadata}\n{r.page_content[:400]}")
