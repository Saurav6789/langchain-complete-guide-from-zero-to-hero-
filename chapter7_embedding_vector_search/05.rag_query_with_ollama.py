from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load embeddings
emb = OllamaEmbeddings(model="mxbai-embed-large")

# Load FAISS index safely
index = FAISS.load_local(
    "faiss_index_ollama", embeddings=emb, allow_dangerous_deserialization=True
)

llm = ChatOllama(model="phi3")

query = "Summarize the performance recommendations in the report."

retrieved = index.similarity_search(query, k=4)

context = "\n\n".join(
    [
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in retrieved
    ]
)

prompt = f"""
You are an expert assistant.

Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer concisely with bullet points.
"""

response = llm.invoke(prompt)
print(response.content)
