import os, asyncio, json
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, SQLiteSession, function_tool

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
    
    #create json file if it doesn't exist
    FILENAME = "memory.json"
    memory_default = {
        "user_profile" : [],
        "order_preferences" : [],
        "other" : []
    }
    
    if not os.path.exists(FILENAME):
        with open(FILENAME, 'w') as f:
            json.dump(memory_default,f, indent=4)
            print(f"Created {FILENAME} with default data")
    else:
        print(f"{FILENAME} already exists")
        

    @function_tool
    def save_memory(memory_type: str, memory: str) -> str:
        """
            Saves memory to a memory  store
        Args:
            memory_type (str): The type of memory to store. Choose between user_profile, order_preferences, or others.
            memory (str): The memory to save
        """
        
        with open(FILENAME, 'r') as f:
            data = json.load(f)
            data[memory_type].append(memory)
            
        with open(FILENAME, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Memory ({memory}) saved")
        return f"Memory ({memory}) saved"
    
    @function_tool
    def load_memory(memory_type : str) -> str:
        """
            Load a set of memory from a memory store
        Args:
            memory_type (str):  The type of memory to store. Choose between user_profile, order_preferences, or others.
        """
        
        with open(FILENAME, 'r') as f:
            data = json.load(f)
        
        return "|".json(data[memory_type])
        
    
    #Create the agent
    agent = Agent(
        name="QuestionAnswer",
        instructions="You are an AI agent that answers questions, You have access to two tools that enable you to save memories and load memories. Save memories when you learn an important fact. Load memories when something is asked for about the user",
        model = model,
        tools=[save_memory,load_memory]
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