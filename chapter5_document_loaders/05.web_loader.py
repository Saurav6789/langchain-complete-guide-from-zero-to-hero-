from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
import os

# Explicit .env path
env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

print("USER_AGENT:", os.getenv("USER_AGENT"))

loader = WebBaseLoader(
    "https://docs.langchain.com/oss/python/integrations/document_loaders"
)

docs = loader.load()

for doc in docs:
    print(doc.page_content[:500])

    print("\nMETADATA:")
    print(doc.metadata)
