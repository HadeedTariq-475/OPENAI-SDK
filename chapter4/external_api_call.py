import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    ModelSettings,
    set_tracing_disabled
)

load_dotenv()
set_tracing_disabled(disabled=True)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)


# Create a tool
@function_tool
def get_price_of_bitcoin() -> str:
    """Get the price of Bitcoin."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    price = response.json()["bitcoin"]["usd"]
    return f"${price:,.2f} USD."
    
        
crypto_agent = Agent(
    name="CryptoTracker",
    instructions="You are a crypto assistant. Use tools to get real-time data.",
    model=model,
    tools=[get_price_of_bitcoin]
    )

result = Runner.run_sync(crypto_agent, "What's the price of Bitcoin?")
print(result.final_output)