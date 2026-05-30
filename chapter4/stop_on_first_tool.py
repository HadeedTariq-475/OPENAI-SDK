import os
from dotenv import load_dotenv
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


@function_tool
def calculate_mortgage(principle_amount : float, annualized_rate: float, number_of_years: int) -> str:
    
    """ This function calculates the mortgage payment
    Args:
        principle_amount : The mortgage amount
        annual_rate : The annualized interest rate in percent form
        years: The loan term in years
    Returns:
        A message stating the monthly payment amount.
    """
    
    monthly_rate = (annualized_rate / 100) / 12
    months = number_of_years * 12
    payment = principle_amount * (monthly_rate) / (1 - (1 + monthly_rate)** - months)
    print(payment)
    return f"${payment:,.2f}."
    
        
agent = Agent(
    name="Mortgage Advisor",
    instructions="You are a mortgage assistant.",
    model=model,
    tools=[calculate_mortgage],
    tool_use_behavior="stop_on_first_tool",
    model_settings=ModelSettings(tool_choice="required")
)

result = Runner.run_sync(
    agent,
    "What is my monthly payments if I borrow $800,000 at 6 percent interest for 30 years?"
)

print(result.final_output)