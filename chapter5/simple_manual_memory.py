import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled

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

#Create the agent
agent = Agent(
    name="QuestionAnswer",
    instructions="You are an AI agent that answers questions",
    model = model
)

#Create an empty list, this is contain the messages
messages = []

#Initial message 
messages.append({"role":"user", "content" : "How hot is the sun"})

#Call the agent
result = Runner.run_sync(agent, messages)
print(result.final_output)

#Append agent response into the messages
messages.append({"role":"assistant", "content" : result.final_output})

# Add a second message
messages.append({"role":"user", "content" : "How big it is?"})

#Again call the agent
result = Runner.run_sync(agent, messages)
print(result.final_output)