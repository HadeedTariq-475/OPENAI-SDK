import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    WebSearchTool,
    CodeInterpreterTool
)

from agents.tool import CodeInterpreter

load_dotenv()
set_tracing_disabled(disabled=True)

web_search_tool = WebSearchTool()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemma-4-26b-a4b-it",
    openai_client=client
)

tool_config = CodeInterpreter(
    container={"type":"auto"},
    type="code_interpreter"
)

code_tool = CodeInterpreterTool(tool_config=tool_config)

location_agent = Agent(
    name = "LocationAgent",
    instructions="You are an AI agent that searches the web and gets latitude and longitude numbers for a particular city",
    model = model,
    tools=[web_search_tool]
)

distance_calculator_agent = Agent(
    name="DistanceCalculatorAgent",
    instructions="You are an AI agent that writes and runs python code to calculate the distance in KM between two latitude/longitude points.",
    model= model,
    tools = [code_tool]
)

agent = Agent(
    name = "Orchestrator",
    instructions="You are an AI agent that calculates the distance between two locations. Use the location agent to get the latitude/longitude. Use the distance calculator agent to calculate the distance.",
    model = model,
    tools=[
        location_agent.as_tool(
            tool_name="LocationAgent",
            tool_description="Returns the latitude and longitude for a particular location"
        ),
        distance_calculator_agent.as_tool(
            tool_name="DistanceCalculatorAgent",
            tool_description="Calculates the distance between two latitude/longitude points"
        )
    ]
)

result = Runner.run_sync(agent, "What's the straight-line distance betweem Toronto and Vancouver?")
print(result.final_output)
