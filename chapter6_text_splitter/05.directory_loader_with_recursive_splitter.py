from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load documents
loader = DirectoryLoader("../data/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_chunks = []
for doc in docs:
    # split_text returns list of strings; use split_documents to preserve metadata if available
    doc_chunks = splitter.split_documents([doc])
    all_chunks.extend(doc_chunks)

print("Total chunks:", len(all_chunks))
print("Chunk example content:", all_chunks[0].page_content[:300])
print("Chunk metadata:", all_chunks[0].metadata)
