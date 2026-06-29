# ============================================================
# LANGGRAPH — STEP 05: Multi-Agent Systems (Supervisor Pattern)
# Goal: Build a system where a supervisor routes tasks to
#       specialized agents, each with their own tools/prompts.
# Run:  python step_05_multi_agent.py
# ============================================================

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Literal
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: WHY MULTI-AGENT? ─────────────────────────────────────────────────
#
# Single agent = one model with ALL tools. Works for simple tasks.
# Multi-agent = specialized agents, each with their own focus.
#
# WHY?
#   • Each agent has a focused system prompt → better at its specialty
#   • Each agent has ONLY relevant tools → less confusion
#   • Can use different models (cheap vs expensive) per agent
#   • Easier to test and debug individual agents
#
# PATTERN: SUPERVISOR
#
#              ┌──────────────────────┐
#              │     SUPERVISOR       │
#              │  (routes to right    │
#              │   specialist)        │
#              └──┬────────┬─────┬───┘
#                 │        │     │
#                 ▼        ▼     ▼
#           ┌────────┐ ┌──────┐ ┌────────┐
#           │RESEARCH│ │ MATH │ │ WRITER │
#           │ AGENT  │ │AGENT │ │ AGENT  │
#           └────────┘ └──────┘ └────────┘
#                 │        │     │
#                 └────┬───┘─────┘
#                      ▼
#              ┌──────────────────────┐
#              │     SUPERVISOR       │ ← decides: done or send to another?
#              └──────────────────────┘


# ── SPECIALIST TOOLS ──────────────────────────────────────────────────────────

# Research tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    results = {
        "python latest": "Python 3.13 released Oct 2024 with improved performance.",
        "ai news": "OpenAI released GPT-4o in 2024. Google launched Gemini 2.0.",
        "langchain": "LangChain v1.0 released with simplified API and middleware.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': Topic is actively discussed."

@tool
def lookup_docs(topic: str) -> str:
    """Look up internal documentation."""
    docs = {
        "api": "Our API supports REST and GraphQL. Rate limit: 1000 req/min.",
        "deployment": "Deploy via CI/CD. Supports AWS, GCP, Azure.",
    }
    for key, value in docs.items():
        if key in topic.lower():
            return value
    return f"No docs found for '{topic}'."

# Math tools
@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"
        return "Invalid expression."
    except Exception as e:
        return f"Error: {e}"

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units (km/miles, kg/lbs, C/F)."""
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("c", "f"): lambda x: x * 9/5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return f"Cannot convert {from_unit} to {to_unit}"


# ── STATE ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str  # Who should act next


# ── SUPERVISOR NODE ───────────────────────────────────────────────────────────

def supervisor(state: State) -> dict:
    """Supervisor decides which specialist handles the request."""
    system = SystemMessage(content="""You are a supervisor routing requests to specialists.

Available agents:
- "researcher": For web search, documentation lookup, factual questions
- "mathematician": For calculations, conversions, math problems
- "writer": For writing tasks — summaries, emails, creative content
- "FINISH": When a specialist has already answered adequately

Look at the conversation. If a specialist already answered, say FINISH.
Respond with ONLY: researcher, mathematician, writer, or FINISH.""")

    response = model.invoke([system] + state["messages"])
    decision = response.content.strip().lower()

    if "researcher" in decision:
        next_agent = "researcher"
    elif "mathematician" in decision or "math" in decision:
        next_agent = "mathematician"
    elif "writer" in decision:
        next_agent = "writer"
    else:
        next_agent = "FINISH"

    print(f"  [Supervisor] → Routing to: {next_agent}")
    return {"next_agent": next_agent}


# ── SPECIALIST NODES ──────────────────────────────────────────────────────────

# Each specialist has its OWN model + tools:
research_model = model.bind_tools([search_web, lookup_docs])
math_model = model.bind_tools([calculate, convert_units])

def researcher(state: State) -> dict:
    """Research agent — finds information."""
    system = SystemMessage(content="You are a research specialist. Use tools to find info. Be concise.")
    response = research_model.invoke([system] + state["messages"])

    # Handle tool calls if any
    if response.tool_calls:
        # Execute tools and get results
        tool_node = ToolNode([search_web, lookup_docs])
        tool_result = tool_node.invoke({"messages": [response]})
        # Get final answer after tool results
        all_msgs = [system] + state["messages"] + [response] + tool_result["messages"]
        final = model.invoke(all_msgs)
        return {"messages": [AIMessage(content=final.content, name="researcher")]}

    return {"messages": [AIMessage(content=response.content, name="researcher")]}

def mathematician(state: State) -> dict:
    """Math agent — does calculations."""
    system = SystemMessage(content="You are a math specialist. Use tools for calculations. Show work.")
    response = math_model.invoke([system] + state["messages"])

    if response.tool_calls:
        tool_node = ToolNode([calculate, convert_units])
        tool_result = tool_node.invoke({"messages": [response]})
        all_msgs = [system] + state["messages"] + [response] + tool_result["messages"]
        final = model.invoke(all_msgs)
        return {"messages": [AIMessage(content=final.content, name="mathematician")]}

    return {"messages": [AIMessage(content=response.content, name="mathematician")]}

def writer(state: State) -> dict:
    """Writer agent — creates content."""
    system = SystemMessage(content="You are a writing specialist. Write clear, professional content.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [AIMessage(content=response.content, name="writer")]}

# ── ROUTING FUNCTION ──────────────────────────────────────────────────────────

def route_supervisor(state: State) -> str:
    """Route based on supervisor's decision."""
    return state["next_agent"]


# ── BUILD THE MULTI-AGENT GRAPH ───────────────────────────────────────────────

graph = StateGraph(State)

# Add all nodes
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("mathematician", mathematician)
graph.add_node("writer", writer)

# Edges
graph.add_edge(START, "supervisor")  # Always start with supervisor

graph.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "mathematician": "mathematician",
    "writer": "writer",
    "FINISH": END,
})

# After each specialist → back to supervisor (to decide: done or more?)
graph.add_edge("researcher", "supervisor")
graph.add_edge("mathematician", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()


# ── RUN THE MULTI-AGENT SYSTEM ────────────────────────────────────────────────

print("=" * 55)
print("QUERY 1: Research question")
print("=" * 55)

result = app.invoke({
    "messages": [HumanMessage(content="What's the latest news about AI?")],
    "next_agent": "",
})
print(f"\nFinal: {result['messages'][-1].content}")


print("\n" + "=" * 55)
print("QUERY 2: Math question")
print("=" * 55)

result = app.invoke({
    "messages": [HumanMessage(content="What is 25 * 47 and convert 100 km to miles?")],
    "next_agent": "",
})
print(f"\nFinal: {result['messages'][-1].content}")


print("\n" + "=" * 55)
print("QUERY 3: Writing task")
print("=" * 55)

result = app.invoke({
    "messages": [HumanMessage(content="Write a professional 2-sentence bio for a Python developer")],
    "next_agent": "",
})
print(f"\nFinal: {result['messages'][-1].content}")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. Supervisor pattern: one router + multiple specialists
# 2. Each specialist has its own system prompt and tools
# 3. model.bind_tools() — give each specialist ONLY its relevant tools
# 4. Supervisor loop: supervisor → specialist → supervisor → ... → END
# 5. Named messages (name="researcher") to track who said what
# 6. Specialist nodes handle their own tool calls internally
# 7. Routing function reads state["next_agent"] to decide path
#
# NEXT: step_06_checkpointing.py — Persistence and state management