from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

load_dotenv()

model = ChatOllama(model="qwen2.5-coder")


def get_stock_price(symbol: str) -> str:
    prices = {"AAPL": "$225.50", "GOOGL": "$172.30", "TSLA": "$248.75"}
    return f"The current price of {symbol} is {prices.get(symbol, 'N/A')}."


def get_stock_news(symbol: str) -> str:
    news = {
        "AAPL": "Apple announces new AI features in iOS 20.",
        "GOOGL": "Google reports strong cloud revenue growth.",
        "TSLA": "Tesla delivers record Q1 vehicles.",
    }
    return f"Latest news for {symbol}: {news.get(symbol, 'No recent news.')}"


tools = [get_stock_price, get_stock_news]

checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "demo-session-1"}}

result1 = agent.invoke(
    {"messages": [HumanMessage(content="What's the price and latest news for AAPL?")]},
    config=config,
)

print("Turn 1:")
print(result1["messages"][-1].content)

result2 = agent.invoke(
    {"messages": [HumanMessage(content="Now what about TSLA?")]},
    config=config,
)

print("\nTurn 2:")
print(result2["messages"][-1].content)
