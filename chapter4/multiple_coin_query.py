import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
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

class Crypto(BaseModel):
    """
    coin_ids: full name string to represent the cryptocurrency
    """
    coin_ids : List[str]

# Create a tool
@function_tool
def get_crypto_prices(crypto: Crypto) -> str:
    """Get the current prices of a list of cryptocurrencies
    Args:
        Crypto: an object with list of coin_ids (e.g. bitcoin, ethereum, litecoin, etc)
    """
    ids = ",".join(crypto.coin_ids)
    url = "https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data
    
        
crypto_agent = Agent(
    name="CryptoTracker",
    instructions="You are a crypto assistant. Use tools to get real-time data. When getting cryptocurrency prices, call the tool only once for all requests.",
    model=model,
    tools=[get_crypto_prices]
    )

result = Runner.run_sync(crypto_agent, "What's the price of Bitcoin? Ethereum?")
print(result.final_output)