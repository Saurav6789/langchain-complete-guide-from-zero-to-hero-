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

# Turn 1
question1 = "What is LangChain?"
messages1 = prompt.format_messages(history=history, question=question1)
response1 = model.invoke(messages1)

print("User:", question1)
print("AI:", response1.content)

history.append(HumanMessage(content=question1))
history.append(AIMessage(content=response1.content))

# Turn 2
question2 = "How is it different from LangGraph?"
messages2 = prompt.format_messages(history=history, question=question2)
response2 = model.invoke(messages2)

print("\nUser:", question2)
print("AI:", response2.content)
