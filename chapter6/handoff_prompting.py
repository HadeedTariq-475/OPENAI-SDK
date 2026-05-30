import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()
set_tracing_disabled(disabled=True)

print(RECOMMENDED_PROMPT_PREFIX)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemma-4-26b-a4b-it",
    openai_client=client
)

complaint_agent = Agent(
    name = "Complaints Agent",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX}. Handle any customer complaints with empathy and clear next steps",
    model=model
)

inquiry_agent = Agent(
    name = "General Inquiry Agent",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX}. Answer general questions about our services promptly",
    model=model
)

triage_agent = Agent(
    name = "Triage Agent",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX}. Triage the user's request and call the appropriate agent",
    model=model,
    handoffs=[complaint_agent,inquiry_agent]
)

while True:
    question = input("You: ")
    result = Runner.run_sync(triage_agent, question)
    print("Agent: ", result.final_output)
