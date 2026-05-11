from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")

summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize this text in 2 bullets:\n\n{text}"
)

answer_prompt = ChatPromptTemplate.from_template(
    "Answer this question directly:\n\n{question}"
)

summarize_chain = summarize_prompt | model | StrOutputParser()
answer_chain = answer_prompt | model | StrOutputParser()


def route_input(data):
    text = data["text"]
    if len(text) > 100:
        return summarize_chain.invoke({"text": text})
    return answer_chain.invoke({"question": text})


router_chain = RunnableLambda(route_input)

print(
    router_chain.invoke(
        {
            "text": "LangChain is a framework used to build LLM applications. It supports chains, tools, agents, memory, and retrieval."
        }
    )
)
print(
    router_chain.invoke(
        {
            "text": "The Costa Brava (Catalan:is a coastal region of Catalonia in northeastern Spain. Sources differ on the exact definition of the Costa Brava. Usually it can be regarded as stretching from the town of Blanes, 60 km (37 mi) northeast of Barcelona, to the French border – in other words it consists of the coast of the province of Girona.In the 1950s, the Costa Brava was identified by the Spanish government and local entrepreneurs as being suitable for substantial development as a holiday destination, mainly for package holiday tourists from Europe. The combination of a very good summer climate, nature, excellent beaches and a favourable foreign exchange rate (before the creation of the single European currency), which made the Costa Brava an attractive tourist destination, was exploited by the construction of large numbers of hotels and apartments in such seaside resorts as Blanes, Tossa de Mar and Lloret de Mar. Tourism rapidly took over from fishing as the principal business of the area."
        }
    )
)
