import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool

#load environment variables from .env file
load_dotenv()
set_tracing_disabled(disabled=True)

# Create client for Gemini (OpenAI format)
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemma-4-26b-a4b-it",
    openai_client=client
)

# Create a tool
@function_tool
def get_order_status(order_id: int) -> str:
    """
    Returns the order status given the order ID
    """
    
    if order_id in (100,101):
        return "Delivered"
    elif order_id in (200,201):
        return "Delayed"
    elif order_id in (300,301):
        return "Cancelled"

#Customer Retention Agent
customer_retention_agent = Agent(
    name = "Customer Retention Agent",
    instructions= "You are an AI Agent that responds to customers that want to close their accounts and retain their business. Be very courteous, relatable and kind. Offer discounts upto 10 percent if it helps",
    model = model
)

#Define the customer service agent
agent = Agent(
    name="Customer Service Agent",
    instructions="You are an AI Agent that helps to respond to customer queries for a local paper company",
    model= model,
    tools = [get_order_status],
    handoffs=[customer_retention_agent]
)

#Run the control logic Framework
result = Runner.run_sync(agent, "I want to cancel my order and account. You delayed the order for the 3rd time!")

#Print the result
print(result.final_output)