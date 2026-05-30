import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
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

complaint_agent = Agent(
    name = "Complaints Agent",
    instructions="Handle any customer complaints with empathy and clear next steps",
    model=model
)

inquiry_agent = Agent(
    name = "General Inquiry Agent",
    instructions="Answer general questions about our services promptly",
    model=model
)

def orchestration(user_message : str):
    
    """ Deterministically delegates requests to the right customer service agent"""
    
    if("complaint" in user_message.lower() or "problem" in user_message.lower()):
        print("Redirecting you to complaints agent")
        chosen_agent = complaint_agent
    else:
        print("Redirecting you to inquiry agent")
        chosen_agent = inquiry_agent
        
    result = Runner.run_sync(chosen_agent, user_message)
    return result.final_output

while True:
    question = input("You: ")
    result = orchestration(question)
    print("Agent: ", result)