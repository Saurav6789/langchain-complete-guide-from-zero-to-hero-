from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(file_path="../data/users.json", jq_schema=".[]", text_content=False)
docs = loader.load()

for doc in docs[:2]:
    print(doc.page_content)
    print(doc.metadata)
