from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../data/learning_langchain.pdf")
docs = loader.load()

for doc in docs[:2]:
    print(doc.page_content[:500])
    print(doc.metadata)
