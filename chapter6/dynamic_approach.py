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

# Select model
model = OpenAIChatCompletionsModel(
    model="gemma-4-26b-a4b-it",
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

triage_agent = Agent(
    name = "Triage Agent",
    instructions="Triage the user's request and call the appropriate agent",
    model=model,
    tools=[
        complaint_agent.as_tool(
            tool_name="ComplaintsAgent",
            tool_description="Introduce yourself as complaints agent. Handle any customer complaints with empathy and clear next steps."
        ),
        inquiry_agent.as_tool(
            tool_name="GeneralInquiryAgent",
            tool_description="Introduce yourself as general inquiry agent. Answer General Questions about our services promptly."
        )
    ]
)

result = Runner.run_sync(triage_agent, "My meal is too hot and how do I get my receipt?")
print(result.final_output)