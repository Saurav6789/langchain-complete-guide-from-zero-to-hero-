from langchain_text_splitters import CharacterTextSplitter

text = open("../data/long_text.txt", "r", encoding="utf-8").read()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks; first chunk preview:\n", chunks[0][:400])
