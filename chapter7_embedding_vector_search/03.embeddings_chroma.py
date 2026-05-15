# embeddings_chroma.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

loader = TextLoader("../data/long_text.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

emb = OllamaEmbeddings(model="mxbai-embed-large")
texts = [d.page_content for d in split_docs]
metadatas = [d.metadata for d in split_docs]

db = Chroma.from_texts(texts, emb, metadatas=metadatas, persist_directory="chroma_db")
db.persist()
# query
results = db.similarity_search("Explain the installation steps", k=4)
for r in results:
    print(r.metadata, r.page_content[:200])
