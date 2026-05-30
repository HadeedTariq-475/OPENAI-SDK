import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    SQLiteSession,
    trace,
    handoff
)
from pydantic import BaseModel

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

sales_agent = Agent(
    name = "Sales Agent",
    instructions="Introduce yourself as the sales agent. Answer general questions about our services promptly",
    model=model
)

triage_agent = Agent(
    name = "Triage Agent",
    instructions="Answer general questions. Triage the user's request and call the appropriate agent",
    model=model,
)

class NameOfAgentToBeHandedOff(BaseModel):
    name_of_agent_to_be_handed_off : str

# Create Logging Function
def log(ctx, name_of_agent):
    msg = f"The system has transfered you to another agent: {name_of_agent.name_of_agent_to_be_handed_off}"
    print(msg)
    
#Create custom handoffs
complaints_handoff = handoff(agent=complaint_agent, on_handoff=log, input_type=NameOfAgentToBeHandedOff)    
sales_handoff = handoff(agent=sales_agent, on_handoff=log, input_type=NameOfAgentToBeHandedOff)
triage_handoff = handoff(agent=triage_agent, on_handoff=log, input_type=NameOfAgentToBeHandedOff)        

    
#handoffs of all agents
complaint_agent.handoffs = [sales_agent,triage_agent]
sales_agent.handoffs= [complaint_agent,triage_agent]
triage_agent.handoffs = [complaint_agent, sales_agent]

#create a session
session = SQLiteSession("first session")
last_agent = triage_agent

with trace("Multi_Agent_System"):
    while True:
        question = input("You: ")
        result = Runner.run_sync(last_agent, question, session=session)
        print("Agent: ", result.final_output)
        last_agent = result.last_agent
