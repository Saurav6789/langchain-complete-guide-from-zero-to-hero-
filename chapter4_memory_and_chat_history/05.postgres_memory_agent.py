from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool  # ✅ Added
import psycopg

load_dotenv()

# ✅ Postgres URI
DB_URI = "postgresql://user:password@localhost:5432/chat_db"

model = ChatOllama(model="qwen2.5-coder")


# ✅ MUST add @tool decorator with docstrings for the agent to work
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

# ✅ PostgresSaver requires a connection pool/context manager
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    # First time: setup tables
    checkpointer.setup()

    # ✅ Agent creation using the loop-enabled create_agent
    agent = create_agent(model, tools, checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-session-456"}}

    print("=== Production Postgres Demo ===")

    # Turn 1
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="What's the price for AAPL?")]},
        config=config,
    )
    print(f"Agent: {result1['messages'][-1].content}")

    # Turn 2
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="What about GOOGL?")]},
        config=config,
    )
    print(f"Agent: {result2['messages'][-1].content}")
