import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    StopAtTools,
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
    
@function_tool
def create_invoice(orderID: int) -> str:
    return f"Invoice for order {orderID}: $123.45 (Generated on 2026-05-24)"

agent = Agent(
    name="Customer Service Agent",
    instructions="You are an AI Agent that helps respond to customer queries for a local paper company.",
    model=model,
    tools=[create_invoice],
    tool_use_behavior= StopAtTools(stop_at_tool_names=["create_invoice"])
)

result = Runner.run_sync(
    agent,
    "Please create an invoice for order 300"
)

print(result.final_output)