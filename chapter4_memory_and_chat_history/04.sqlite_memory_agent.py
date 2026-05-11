from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent  # ✅ Correct import for v1.0+
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOllama(model="llama3.1")


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a given symbol like AAPL, GOOGL, or TSLA."""
    prices = {"AAPL": "$225.50", "GOOGL": "$172.30", "TSLA": "$248.75"}
    return f"The current price of {symbol} is {prices.get(symbol, 'N/A')}."


@tool
def get_stock_news(symbol: str) -> str:
    """Get latest news for a stock symbol like AAPL, GOOGL, or TSLA."""
    news = {
        "AAPL": "Apple announces new AI features in iOS 20.",
        "GOOGL": "Google reports strong cloud revenue growth.",
        "TSLA": "Tesla delivers record Q1 vehicles.",
    }
    return f"Latest news for {symbol}: {news.get(symbol, 'No recent news.')}"


tools = [get_stock_price, get_stock_news]

# ✅ In v1.0, create_agent takes the model and tools.
# It handles the loop (call tool -> get result -> call model again) automatically.
with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
    agent = create_agent(model, tools, checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-session-1"}}

    print("=== Turn 1 ===")
    # Using 'messages' key is the standard way to interact with graph-based agents
    result1 = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What's the price and latest news for AAPL?")
            ]
        },
        config=config,
    )
    # Print the last message (the conversational final answer)
    print(result1["messages"][-1].content)

    print("\n=== Turn 2 (Memory persists!) ===")
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="Now what about TSLA?")]},
        config=config,
    )
    print(result2["messages"][-1].content)
