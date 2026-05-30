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
    instructions="You are an AI agent that answers questions, give me concrete and concise answers.",
    model = model
)

#Create an empty list, this is contain the messages
messages = []

while True:
    question = input("You: ")
    messages.append({"role":"user","content":question})
    result = Runner.run_sync(agent, messages)
    print(result.final_output)
    messages = result.to_input_list()