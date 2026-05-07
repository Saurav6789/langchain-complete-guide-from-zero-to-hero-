from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")
hardcoded_prompt = model.invoke("List 3 benefits of using langchain")
print("Hard Coded Prompt", hardcoded_prompt.content)
