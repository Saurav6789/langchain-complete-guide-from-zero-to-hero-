from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("../data/structured_text.txt", "r", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],  # paragraph, line, sentence, word, char
)
chunks = splitter.split_text(text)
print(len(chunks))
for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} ---")
    print(chunk[:400])
    print()
