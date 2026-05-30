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

# Create our agents
landlord_agent = Agent(
    name="Landlord Agent",
    instructions="Argue against rent control from the perspective of a landlord. Present strong economic and property rights arguments.",
    model=model
)

tenant_agent = Agent(
    name="Tenant Agent",
    instructions="Argue in favor of rent control from the perspective of a tenant. Emphasize affordability, housing rights, and tenant protections.",
    model=model
)

summarizer_agent = Agent(
    name="Summarizer Agent",
    instructions="Summarize the main arguments presented by both the landlord and tenant agents in a neutral and concise way.",
    model=model
)


# Create a session
session = SQLiteSession("decentralized")
landlord_turn = True
conversation_history = []

with trace("Decentralized system"):
    print("Topic: Should there be rent control?\n")
        
    for _ in range(6): # 6 rounds of back and forth
        if landlord_turn:
            agent = landlord_agent
        else:
            agent = tenant_agent
                
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            
        response = Runner.run_sync(
            agent,
            prompt or "Debate starting now.", 
            session=session
        )
            
        print(f"{agent.name}:\n{response.final_output}\n")
        conversation_history.append({"role": agent.name, "content": response.final_output})
            
        landlord_turn = not landlord_turn

    # After the debate, have the moderator summarize
    summary_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    result = Runner.run_sync(summarizer_agent, summary_prompt, session=session)
        
    print("================================")
    print("Summary of the Debate:")
    print(result.final_output)