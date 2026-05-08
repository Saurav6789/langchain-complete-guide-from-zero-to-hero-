from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()
model = ChatOllama(model="qwen2.5-coder")

print("=== FewShotPromptTemplate Demo ===\n")

# Classic Q&A examples
qa_examples = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level programming language known for readability.",
    },
    {
        "question": "What is JavaScript?",
        "answer": "JavaScript is a language primarily used for web development.",
    },
]

qa_example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Q: {question}\nA: {answer}\n"
)

qa_prompt = FewShotPromptTemplate(
    examples=qa_examples,
    example_prompt=qa_example_prompt,
    suffix="Q: {input}\nA:",
    input_variables=["input"],
)

print("1️⃣ Q&A Style:")
print(model.invoke(qa_prompt.format(input="What is LangChain?")).content)
print()

# Classification examples
classification_examples = [
    {"text": "I love this!", "sentiment": "positive"},
    {"text": "This is terrible.", "sentiment": "negative"},
    {"text": "Pretty good overall.", "sentiment": "positive"},
]

classification_prompt = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}\n",
)

fewshot_classifier = FewShotPromptTemplate(
    examples=classification_examples,
    example_prompt=classification_prompt,
    suffix="Text: {input}\nSentiment:",
    input_variables=["input"],
)

print("2️⃣ Sentiment Classifier:")
print(model.invoke(fewshot_classifier.format(input="Amazing product!")).content)
print()

print("✅ Few-shot examples teach the model the pattern!")
