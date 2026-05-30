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

# Create our specialized leaf agents
# Specialized science agents
physics_agent = Agent(
    name="Physics Agent", 
    instructions="Answer questions about physics.",
    model=model
)
chemistry_agent = Agent(
    name="Chemistry Agent", 
    instructions="Answer questions about chemistry.",
    model=model
)
medical_agent = Agent(
    name="Medical Agent", 
    instructions="Answer questions about medical science.",
    model=model
)

# Specialized history agents
politics_agent = Agent(
    name="Politics Agent", 
    instructions="Answer questions about political history.",
    model=model
)
warfare_agent = Agent(
    name="Warfare Agent", 
    instructions="Answer questions about wars and military history.",
    model=model
)
culture_agent = Agent(
    name="Culture Agent", 
    instructions="Answer questions about cultural history.",
    model=model
)

# Manager agents with handoffs to their respective domains
science_manager = Agent(
    name="Science Manager",
    instructions="Manage science related queries and route them to the appropriate subdomain agent.",
    model=model,
    handoffs=[physics_agent, chemistry_agent, medical_agent]
)

history_manager = Agent(
    name="History Manager",
    instructions="Manage history related queries and route them to the appropriate subdomain agent.",
    model=model,
    handoffs=[politics_agent, warfare_agent, culture_agent]
)

# Top-level triage agent
triage_agent = Agent(
    name="Research Triage Agent",
    instructions="Triage the user's question and decide whether it's science or history related, and route accordingly.",
    model=model,
    handoffs=[science_manager, history_manager]
)


# Create a session
session = SQLiteSession("hierarchy")
last_agent = triage_agent
    
with trace("Hierarchical system"):
    while True:
        try:
            question = input("You: ")
            if question.lower() in ['exit', 'quit']:
                break
                    
            result = Runner.run_sync(last_agent, question, session=session)
            print(f"Agent: {result.final_output}")
                
            # Update the last_agent so the conversation continues in the correct context
            last_agent = result.last_agent
                
        except KeyboardInterrupt:
            break