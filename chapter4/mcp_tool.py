import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    HostedMCPTool,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled
)
from agents.tool import Mcp

load_dotenv()
set_tracing_disabled(disabled=True)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemma-4-26b-a4b-it",
    openai_client=client
)

#Create tool
tool_config = Mcp(
    server_label="CryptocurrencyPriceFetcher",
    server_url="https://mcp.api.coingecko.com/sse",
    type="mcp",
    require_approval="never"
)

mcp_tool = HostedMCPTool(tool_config=tool_config)

agent = Agent(
    name = "Crypto Agent",
    instructions="You are an AI agent that returns crypto prices.",
    model=model,
    tools=[mcp_tool]
)

result = Runner.run_sync(agent, "What is the price of bitcoin?")
print(result.final_output)
