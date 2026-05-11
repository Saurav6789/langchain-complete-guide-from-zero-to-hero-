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
            "You are a helpful assistant. Use the conversation history to answer follow-up questions.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

history = []


def chat(question):
    global history
    messages = prompt.format_messages(history=history, question=question)
    response = model.invoke(messages)
    print("User:", question)
    print("AI:", response.content)
    print()
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))


chat("What is LangChain?")
chat("How is it different from LangGraph?")
chat("Which one is better for building workflows?")
