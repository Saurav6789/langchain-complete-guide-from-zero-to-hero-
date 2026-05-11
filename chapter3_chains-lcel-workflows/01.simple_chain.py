from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words for a beginner."
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic": "LangChain chains"})
print(result)
