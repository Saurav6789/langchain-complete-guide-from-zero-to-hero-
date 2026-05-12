from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("../data/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

print(f"Loaded {len(docs)} documents")
