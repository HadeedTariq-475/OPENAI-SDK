import os
from dotenv import load_dotenv
from pydantic import BaseModel
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

class RefundRequest(BaseModel):
    order_id : str
    customer_email : str
    reason : str
    
@function_tool
def process_refund(request: RefundRequest) -> str:
    
    """ Process the refund requests and return confirmation."""
    
    return (f"refund request for order {request.order_id} has been submitted."
            f"A confirmation will be sent to {request.customer_email}."
            )
        
agent = Agent(
    name="Customer Service Agent",
    instructions="You are an AI Agent that helps respond to customer queries for a local paper company.",
    model=model,
    tools=[process_refund],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["process_refund"])
)

result = Runner.run_sync(
    agent,
    "I want a refund for order 101."
)

print(result.final_output)