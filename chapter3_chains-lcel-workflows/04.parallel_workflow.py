from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")

main_idea_prompt = ChatPromptTemplate.from_template(
    "Extract the main idea from this text:\n\n{text}"
)

action_items_prompt = ChatPromptTemplate.from_template(
    "Extract action items from this text:\n\n{text}"
)

main_idea_chain = main_idea_prompt | model | StrOutputParser()
action_items_chain = action_items_prompt | model | StrOutputParser()

parallel_chain = RunnableParallel(
    main_idea=main_idea_chain,
    action_items=action_items_chain,
)

result = parallel_chain.invoke(
    {
        "text": "LangChain helps build AI apps. Start with prompts, then add chains, then memory, retrieval, and tools."
    }
)

print(result["main_idea"])
print("\n---\n")
print(result["action_items"])
