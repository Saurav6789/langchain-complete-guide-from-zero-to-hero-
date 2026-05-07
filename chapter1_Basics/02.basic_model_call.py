from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Read values from .env
model_name = os.getenv("MODEL_NAME")
system_prompt = os.getenv("SYSTEM_PROMPT")

# Initialize model using env variable
model = ChatOllama(model=model_name)

# Create messages using env-based system prompt
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="Explain what LangChain is in 3 bullet points."),
]

# Invoke model
response = model.invoke(messages)

# Print output
print(response.content)
