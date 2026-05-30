# OpenAI Agents SDK Learning Examples

This repository contains small, focused examples built while learning the OpenAI Agents SDK. The examples use `openai-agents` with an OpenAI compatible Gemini endpoint, and show how to build single agents, agents with tools, agents with memory, and multi agent systems.

## Setup

* Python: `>=3.12`
* Main dependencies: `openai-agents`, `python-dotenv`, `requests`, `pydantic`
* Environment variable required: `GEMINI_API_KEY`

Most files create an `AsyncOpenAI` client with:

```python
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
```

We create this custom `AsyncOpenAI` client because these examples use a Gemini API key with Gemini's OpenAI compatible endpoint. If you are using an OpenAI API key directly, you usually do not need to create a custom client or set a custom `base_url`; the SDK can use the default OpenAI client configuration.

For RAG, you can also use OpenAI's proprietary vector store directly instead of building your own retrieval storage from scratch.

Run any example directly, for example:

```bash
uv run python chapter5/long_term_basic_persistant_memory.py
```

## Chapter 3: Basic Agent + Tool + Handoff

**File:** `chapter3/customer_service_agent.py`

Concepts learned:

* Creating an `Agent`
* Running an agent with `Runner.run_sync`
* Adding a Python function as a tool using `@function_tool`
* Creating a handoff from one agent to another

Agents built:

* `Customer Service Agent`: answers customer queries for a local paper company and can check order status.
* `Customer Retention Agent`: receives account cancellation cases and tries to retain the customer politely.

Tool built:

* `get_order_status(order_id)`: returns dummy statuses like delivered, delayed, or cancelled.

## Chapter 4: Tools and Tool Control

Chapter 4 focuses on giving agents external abilities through tools.

### Function Tools

**Files:** `model_settings.py`, `stop_at_tools.py`, `stop_on_first_tool.py`, `chained_tools_call.py`

Concepts learned:

* Creating tools with `@function_tool`
* Overriding tool names and descriptions
* Forcing tool use with `ModelSettings(tool_choice="required")`
* Stopping after a tool call with `stop_on_first_tool`
* Stopping at selected tools with `StopAtTools`
* Chaining multiple tools in one task

Agents built:

* `Customer Service Agent`: checks order status and creates invoices.
* `Mortgage Advisor`: calculates monthly mortgage payments.
* `CustomerSupportAgent`: gets a customer's orders, then checks each order status.

Tools built:

* `get_status_of_current_order`
* `create_invoice`
* `calculate_mortgage`
* `get_customer_orders`
* `get_order_information`

### Structured Tool Inputs with Pydantic

**Files:** `complex_tool_input_with_pydantic.py`, `database_query_tool.py`, `multiple_coin_query.py`

Concepts learned:

* Defining structured tool schemas with `pydantic.BaseModel`
* Passing complex objects into tools
* Querying simulated data stores
* Handling list inputs in a single tool call

Agents built:

* `Customer Service Agent`: processes refund requests.
* `SupportHelper`: fetches support ticket history from a simulated database.
* `CryptoTracker`: fetches one or more cryptocurrency prices.

Models/tools built:

* `RefundRequest`
* `CustomerQuery`
* `Crypto`
* `process_refund`
* `get_customer_tickets`
* `get_crypto_prices`

### External APIs, Hosted Tools, and Agents as Tools

**Files:** `external_api_call.py`, `mcp_tool.py`, `agents_as_tool.py`

Concepts learned:

* Calling external APIs from custom tools
* Using a hosted MCP tool
* Using `WebSearchTool`
* Using `CodeInterpreterTool`
* Turning agents into tools with `agent.as_tool(...)`
* Building an orchestrator agent that delegates subtasks

Agents built:

* `CryptoTracker`: calls CoinGecko through a custom API tool.
* `Crypto Agent`: uses a hosted CoinGecko MCP server.
* `LocationAgent`: searches for latitude and longitude.
* `DistanceCalculatorAgent`: calculates distance using code execution.
* `Orchestrator`: combines the location and distance agents to answer distance questions.

## Chapter 5: Memory and Conversation State

Chapter 5 explores short term and long term memory patterns.

### Manual Conversation Memory

**Files:** `simple_manual_memory.py`, `to_input_list.py`

Concepts learned:

* Passing a list of messages into `Runner`
* Appending user and assistant messages manually
* Reusing `result.to_input_list()` to continue a conversation

Agent built:

* `QuestionAnswer`: answers follow up questions using prior conversation messages.

### Sliding Window Memory

**File:** `sliding_window.py`

Concepts learned:

* Using `collections.deque(maxlen=...)` to keep only recent messages
* Preventing context from growing forever
* Running agents asynchronously with `await Runner.run(...)`

Agent built:

* `QuestionAnswer`: answers with only the latest window of conversation history.

### Sessions and Persistent Memory

**Files:** `conversations_with_sessions.py`, `long_term_basic_persistant_memory.py`

Concepts learned:

* Using `SQLiteSession` to store conversation history
* Keeping state across turns without manually passing message lists
* Persisting session data to `messages.db`
* Using async loops for long running chat sessions

Agent built:

* `QuestionAnswer`: a conversational Q&A agent backed by SDK sessions.

### Tool Based Structured Memory

**File:** `tool_based_structured_memory.py`

Concepts learned:

* Combining `SQLiteSession` conversation memory with a separate structured memory store
* Saving important facts to `memory.json`
* Loading memories by category through tools
* Giving the agent explicit save and load memory abilities

Agent built:

* `QuestionAnswer`: can save and retrieve facts about the user.

Tools built:

* `save_memory(memory_type, memory)`
* `load_memory(memory_type)`

Memory categories:

* `user_profile`
* `order_preferences`
* `other`

## Chapter 6: Multi Agent Systems and Handoffs

Chapter 6 explores different ways to coordinate multiple agents.

### Basic Handoff

**File:** `basic_handoff.py`

Concepts learned:

* Creating specialist agents
* Giving a triage agent a list of `handoffs`
* Letting the model decide which specialist should respond

Agents built:

* `Triage Agent`
* `Complaints Agent`
* `General Inquiry Agent`

### Deterministic vs Dynamic Orchestration

**Files:** `deterministic_approach.py`, `dynamic_approach.py`

Concepts learned:

* Routing with normal Python control flow
* Routing dynamically by exposing agents as tools
* Comparing rule based orchestration with model driven delegation

Agents built:

* `Complaints Agent`
* `General Inquiry Agent`
* `Triage Agent`

### Handoff Prompting and Customization

**Files:** `handoff_prompting.py`, `handoff_customization.py`

Concepts learned:

* Using `RECOMMENDED_PROMPT_PREFIX` for better handoff behavior
* Creating custom handoffs with `handoff(...)`
* Adding `on_handoff` logging
* Using Pydantic input for handoff metadata
* Tracking `result.last_agent` so the conversation continues with the active specialist

Agents built:

* `Triage Agent`
* `Complaints Agent`
* `Sales Agent`
* `General Inquiry Agent`

### Multi Agent Switching

**File:** `multi_agent_switching.py`

Concepts learned:

* Allowing agents to hand off to each other
* Maintaining the current active agent with `last_agent`
* Keeping multi turn state with `SQLiteSession`
* Tracing a multi agent conversation with `trace(...)`

Agents built:

* `Triage Agent`
* `Complaints Agent`
* `Sales Agent`

### Hierarchical System

**File:** `hierarchical _system.py`

Concepts learned:

* Building a hierarchy of manager agents and specialist agents
* Routing first by broad domain, then by subdomain
* Continuing conversation with the last active agent

Agents built:

* `Research Triage Agent`
* `Science Manager`
* `History Manager`
* `Physics Agent`
* `Chemistry Agent`
* `Medical Agent`
* `Politics Agent`
* `Warfare Agent`
* `Culture Agent`

### Decentralized System

**File:** `decentrailized_system.py`

Concepts learned:

* Simulating a back and forth debate between agents
* Maintaining shared conversation history
* Summarizing multi agent output with a separate agent

Agents built:

* `Landlord Agent`
* `Tenant Agent`
* `Summarizer Agent`

### Swarm System

**File:** `swarm_system.py`

Concepts learned:

* Running many agents with different roles on the same prompt
* Collecting their outputs
* Aggregating many responses into one final answer

Agents built:

* Role agents: Urban Planner, Artist, Chef, Engineer, Teacher, Doctor, Mechanic, Lawyer, Historian, Environmentalist
* `City Design Aggregator`: combines all role agent ideas into one city plan.

## Key Takeaways

* An `Agent` combines instructions, a model, tools, and optional handoffs.
* `Runner.run_sync(...)` is useful for simple scripts; `await Runner.run(...)` is useful for async apps.
* Tools let agents access code, APIs, databases, MCP servers, web search, and code execution.
* Pydantic models make tool inputs structured and reliable.
* Sessions such as `SQLiteSession` provide conversation memory without manually managing message lists.
* Separate memory stores like `memory.json` are useful for structured long term facts.
* Handoffs, agents as tools, and orchestration logic allow multiple agents to collaborate.
