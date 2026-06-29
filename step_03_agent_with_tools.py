# ============================================================
# LANGGRAPH — STEP 03: Agent with Tools (ReAct Pattern)
# Goal: Build a tool-calling agent from scratch using LangGraph.
#       Understand the loop: model → [tool calls?] → tools → model → ...
# Run:  python step_03_agent_with_tools.py
# ============================================================

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# add_messages is a smart reducer:
#   - Appends new messages to the list
#   - Handles deduplication by message ID
#   - Properly merges tool call results

from langgraph.prebuilt import ToolNode
# ToolNode = prebuilt node that executes tool calls from AIMessage

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── THE AGENT LOOP (ReAct Pattern) ────────────────────────────────────────────
#
#   START → model_node → [has tool_calls?]
#                           ├── YES → tool_node → model_node (LOOP)
#                           └── NO  → END (final answer)
#
# This is EXACTLY what create_agent() does internally.
# Here we build it manually to understand the mechanics.


# ── DEFINE TOOLS ──────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {
        "london": "Cloudy, 15°C, humidity 80%",
        "tokyo": "Sunny, 25°C, humidity 45%",
        "paris": "Rainy, 12°C, humidity 90%",
        "new york": "Clear, 20°C, humidity 55%",
    }
    return weather.get(city.lower(), f"No weather data for {city}")

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: '2 + 2', '10 * 5'."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"Result: {eval(expression)}"
        return "Error: Invalid characters"
    except Exception as e:
        return f"Error: {e}"

@tool
def search_knowledge(query: str) -> str:
    """Search for information about a topic."""
    knowledge = {
        "python": "Python is a high-level language created in 1991 by Guido van Rossum.",
        "langgraph": "LangGraph is a library for building stateful agent workflows as graphs.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"No specific info found for: {query}"

tools = [get_weather, calculate, search_knowledge]


# ── BUILD THE AGENT GRAPH ─────────────────────────────────────────────────────

# Step 1: Define State (messages with add_messages reducer)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Annotated[list, add_messages] means:
    #   When a node returns {"messages": [new_msg]},
    #   it APPENDS to the list (doesn't replace).

# Step 2: Bind tools to model
model_with_tools = model.bind_tools(tools)
# bind_tools() tells the model about available tools (their schemas).
# Model can then decide to call them by returning tool_calls in its response.

# Step 3: Define the model node
def call_model(state: AgentState) -> dict:
    """Call the LLM. It may request tool calls or give a final answer."""
    system = SystemMessage(content="You are helpful. Use tools when needed. Be concise.")
    messages = [system] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Step 4: Define the routing function
def should_continue(state: AgentState) -> str:
    """Check if model wants to call tools or is done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"    # Model wants to use tools → go to ToolNode
    return "end"          # No tool calls → final answer, go to END

# Step 5: Create ToolNode
tool_node = ToolNode(tools)
# ToolNode automatically:
#   1. Reads tool_calls from the last AIMessage
#   2. Executes each requested function
#   3. Returns ToolMessage(s) with results

# Step 6: Build the graph
graph = StateGraph(AgentState)

graph.add_node("model", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, {
    "tools": "tools",   # Model wants tools → go to tool_node
    "end": END,         # Model is done → finish
})
graph.add_edge("tools", "model")  # After tools → back to model (THE LOOP)

agent = graph.compile()


# ── RUN THE AGENT ─────────────────────────────────────────────────────────────

print("=" * 55)
print("QUERY 1: Simple tool use")
print("=" * 55)

from langchain.messages import HumanMessage

result = agent.invoke({"messages": [HumanMessage(content="What's the weather in Tokyo?")]})
print(f"Answer: {result['messages'][-1].content}")


print("\n" + "=" * 55)
print("QUERY 2: Multi-tool (agent chains tools)")
print("=" * 55)

result = agent.invoke({
    "messages": [HumanMessage(content="What's 25 * 17 and the weather in London?")]
})
print(f"Answer: {result['messages'][-1].content}")


print("\n" + "=" * 55)
print("QUERY 3: No tools needed")
print("=" * 55)

result = agent.invoke({
    "messages": [HumanMessage(content="What color is the sky?")]
})
print(f"Answer: {result['messages'][-1].content}")


# ── VIEW INTERNAL STEPS ───────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("INTERNAL STEPS (full message trace)")
print("=" * 55)

result = agent.invoke({
    "messages": [HumanMessage(content="Weather in Paris and what is 100/4?")]
})

for msg in result["messages"]:
    msg_type = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  [{msg_type}] Calls: {[tc['name'] for tc in msg.tool_calls]}")
    elif hasattr(msg, "name") and msg.name:
        print(f"  [Tool '{msg.name}'] → {msg.content}")
    else:
        print(f"  [{msg_type}] {msg.content[:100]}")


# ── ALTERNATIVE: tools_condition (prebuilt routing) ───────────────────────────
#
# Instead of writing should_continue() manually, use:
#
#   from langgraph.prebuilt import tools_condition
#   graph.add_conditional_edges("model", tools_condition)
#
# tools_condition does exactly what should_continue does:
#   has tool_calls → "tools"
#   no tool_calls → "__end__"


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. AgentState with Annotated[list, add_messages] — append-only messages
# 2. model.bind_tools(tools) — tell model about available tools
# 3. ToolNode(tools) — prebuilt node that executes tool calls
# 4. The loop: model → [tool_calls?] → tools → model → ... → END
# 5. should_continue() routes based on tool_calls presence
# 6. tools_condition — prebuilt alternative to manual routing
# 7. This is EXACTLY what create_agent() builds internally!
#
# NEXT: step_04_human_in_the_loop.py — Pause for human approval