import os, asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, SQLiteSession

#load environment variables from .env file
load_dotenv()
set_tracing_disabled(disabled=True)

async def main():
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

    # Create Session
    session = SQLiteSession("first session", db_path="messages.db")

    while True:
        question = input("You: ")
        result  = await Runner.run(agent,question,session=session)
        print("Agent: ", result.final_output)
        
# Execute the async main function
if __name__ == "__main__":
    asyncio.run(main())