import os
from dotenv import load_dotenv
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

@function_tool(
    name_override="get_status_of_current_order",
    description_override="Returns the status of an order given the customer's Order ID"
)
def get_order_status(order_id: int) -> str:
    """
    Returns the order status given the order ID.

    Args:
        order_id: Order ID of the customer's order.

    Returns:
        Status message of the customer's order.
    """

    if order_id in (100, 101):
        return "Delivered"
    elif order_id in (200, 201):
        return "Delayed"
    elif order_id in (300, 301):
        return "Cancelled"
    else:
        return "Order not found"
    
agent = Agent(
    name="Customer Service Agent",
    instructions="You are an AI Agent that helps respond to customer queries for a local paper company.",
    model=model,
    tools=[get_order_status],
    model_settings=ModelSettings(tool_choice="required")
)

result = Runner.run_sync(
    agent,
    "Can you check the status of order 101?"
)

print(result.final_output)