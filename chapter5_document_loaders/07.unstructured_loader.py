from langchain_community.document_loaders import UnstructuredImageLoader

loader = UnstructuredImageLoader("../data/Image.png")
docs = loader.load()

for doc in docs[:2]:
    print(doc.page_content[:500])
    print(doc.metadata)
