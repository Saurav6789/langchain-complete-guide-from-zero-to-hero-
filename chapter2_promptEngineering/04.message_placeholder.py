from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Compare frameworks clearly and concisely.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

history = [
    HumanMessage(content="What is LangChain?"),
    AIMessage(
        content="LangChain is a framework for building LLM applications with prompts, tools, retrieval, and agents."
    ),
    HumanMessage(content="What is LangGraph?"),
    AIMessage(
        content="LangGraph is a framework for building stateful, graph-based agent workflows with explicit control over steps and memory."
    ),
]

messages = prompt.format_messages(
    history=history, question="How do LangChain and LangGraph compare?"
)

response = model.invoke(messages)
print(response.content)
