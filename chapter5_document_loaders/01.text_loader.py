from langchain_community.document_loaders import TextLoader

loader = TextLoader("..\data\example.txt", encoding="utf-8")
docs = loader.load()

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
