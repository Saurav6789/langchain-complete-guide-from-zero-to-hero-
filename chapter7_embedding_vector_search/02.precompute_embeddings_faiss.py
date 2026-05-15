from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ---- CONFIG ----
PDF_PATH = "../data/learning_langchain.pdf"
MODEL_NAME = "mxbai-embed-large"  # use an embeddings-capable Ollama model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 32

# ---- 1) Load PDF ----
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages/documents")

# ---- 2) Split into chunks ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
split_docs = splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks")

# ---- 3) Create embeddings model ----
emb = OllamaEmbeddings(model=MODEL_NAME)

# quick check
test_vec = emb.embed_query("test")
print(f"Embedding dimension: {len(test_vec)}")

# ---- 4) Build FAISS index ----
texts = [d.page_content for d in split_docs]
metadatas = [d.metadata for d in split_docs]

index = FAISS.from_texts(texts, emb, metadatas=metadatas)

# ---- 5) Save index ----
index.save_local("faiss_index_ollama")
print("FAISS index saved to faiss_index_ollama/")
