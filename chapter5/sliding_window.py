import os
import asyncio
from dotenv import load_dotenv
from collections import deque
from agents import Agent, Runner, OpenAIChatCompletionsModel,AsyncOpenAI , set_tracing_disabled
load_dotenv()
set_tracing_disabled(disabled=True)

async def main():
    # Create client inside the async context
    client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemma-4-26b-a4b-it",
        openai_client=client
    )

    agent = Agent(
        name="QuestionAnswer",
        instructions="You are an AI agent that answers questions, give me concrete and concise answers.",
        model=model
    )

    WINDOW_SIZE = 5
    messages = deque(maxlen=WINDOW_SIZE)

    while True:
        question = input("You: ")
        messages.append({"role": "user", "content": question})
        
        # Use await Runner.run instead of run_sync
        result = await Runner.run(agent, list(messages))
        
        print(result.final_output)
        messages.append({"role": "assistant", "content": result.final_output})

# Execute the async main function
if __name__ == "__main__":
    asyncio.run(main())