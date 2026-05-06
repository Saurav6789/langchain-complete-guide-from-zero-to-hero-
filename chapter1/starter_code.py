from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    prices = {"AAPL": "$276.83", "GOOGL": "$172.30", "TSLA": "$248.75"}
    return f"The current price of {symbol} is {prices.get(symbol, 'N/A')}."


@tool
def get_stock_news(symbol: str) -> str:
    """Get latest news for a stock symbol."""
    news = {
        "AAPL": "Apple updated record revenue in Q1 2026 and announced new AI features.",
        "GOOGL": "Google reports strong cloud revenue growth.",
        "TSLA": "Tesla delivers record Q1 vehicles.",
    }
    return f"Latest news for {symbol}: {news.get(symbol, 'No recent news.')}"


# Direct model call with tools + clear instructions
model = ChatOllama(model="qwen2.5-coder")

# Execute tools for demo clarity
price = get_stock_price.invoke({"symbol": "AAPL"})
news = get_stock_news.invoke({"symbol": "AAPL"})

messages = [HumanMessage(content=f"""\nUse this tool data to answer the user question:

Stock Price Tool Result: {price}
Stock News Tool Result: {news}

User Question: What's the price and latest news for AAPL?

Provide a single, clear paragraph answer. No JSON or tool calls needed now.""")]

result = model.invoke(messages)

print("🧠 Agent Final Answer:")
print(result.content)
