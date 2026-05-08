from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# Initialize model
model = ChatOllama(model="qwen2.5-coder")
role = "You are a fitness coach"
question = "Create a 3-month plan for flat abs"
context = """
    Focus on:
    - Simple beginner exercises
    - Proper rest days
    - Short workout duration (30–45 minutes)
    - Motivation and consistency tips
    - Avoid advanced gym terminology
"""

# Create chat prompt template
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly experienced {role} expert who gives practical, beginner-friendly advice.",
        ),
        ("user", "{question}"),
        ("assistant", "{context}"),
    ]
)

# Format messages
messages = chat_template.format_messages(role=role, question=question, context=context)

# Invoke model
workout = model.invoke(messages)

# Print response
print("\n🔵 Chat Template Response:\n")
print(workout.content)
