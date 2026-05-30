import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled
)

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

# Define the first tool to get all orders for a given customer
@function_tool
def get_customer_orders(customer_id: str) -> str:
    """
    Retrieve all order IDs associated with a given customer ID.
    Args:
        customer_id: the customer ID
    """
    # Dummy implementation
    if customer_id == "CUST123":
        return ["ORD001", "ORD002", "ORD003"]

# Define the second tool to get status of a specific order
@function_tool
def get_order_information(order_id: str) -> str:
    """
    Fetch detailed information about a specific order.
    """
    # Dummy implementation
    status_map = {
        "ORD001": "Shipped",
        "ORD002": "Processing",
        "ORD003": "Delivered"
    }
    return f"Order {order_id} is currently {status_map.get(order_id, 'Unknown')}."

# Define the agent
customer_service_agent = Agent(
    name="CustomerSupportAgent",
    instructions="You are a customer service assistant.",
    model = model,
    tools=[get_customer_orders, get_order_information] 
)

# Run the agent
result = Runner.run_sync(
    customer_service_agent, 
    "Please check the status of my orders? My customer ID is CUST123."
)
print(result.final_output)