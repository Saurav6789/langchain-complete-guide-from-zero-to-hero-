from langchain_text_splitters import TokenTextSplitter

# Load text
with open("../data/token_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create splitter
splitter = TokenTextSplitter(
    chunk_size=100, chunk_overlap=20  # tokens per chunk  # overlap in tokens
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
