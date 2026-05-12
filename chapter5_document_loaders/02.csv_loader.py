from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="../data/tmdb_5000_movies.csv", encoding="utf-8")

docs = loader.load()

for doc in docs[:5]:
    print(doc.page_content)
    print(doc.metadata)
