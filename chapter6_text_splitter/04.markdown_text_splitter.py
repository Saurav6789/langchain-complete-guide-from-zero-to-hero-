from langchain_text_splitters import MarkdownTextSplitter

md = open("../data/guide.md", "r", encoding="utf-8").read()
md_splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)
md_chunks = md_splitter.split_text(md)
for c in md_chunks[:3]:
    print("---\n", c[:400])
