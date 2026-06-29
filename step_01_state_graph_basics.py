# ============================================================
# LANGGRAPH — STEP 01: StateGraph Basics
# Goal: Understand the core building blocks — State, Nodes, Edges.
#       Build a simple linear graph: research → summarize → end.
# Run:  python step_01_state_graph_basics.py
# ============================================================

# ── INSTALL ──────────────────────────────────────────────────────────────────
# pip install -U langgraph "langchain[openai]" python-dotenv

from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# StateGraph — the builder for creating graph-based workflows
# START — special node: where the graph begins
# END — special node: where the graph finishes

from langchain.chat_models import init_chat_model

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: WHY LANGGRAPH? ───────────────────────────────────────────────────
#
# LangChain chains: A → B → C → Done (one pass, no loops)
# LangGraph:        A → B → [Check] → back to A or → C → Done
#
# LangGraph adds:
#   ✅ Conditional routing (if/else at runtime)
#   ✅ Cycles/loops (retry, iterate until good enough)
#   ✅ State management (data flows through the whole graph)
#   ✅ Human-in-the-loop (pause for approval)
#   ✅ Checkpointing (resume after failure)
#   ✅ Multi-agent coordination
#
# Think of LangGraph as a FLOWCHART your AI follows.


# ── CORE BUILDING BLOCKS ──────────────────────────────────────────────────────
#
# 1. STATE   = TypedDict that flows through the graph (shared data)
# 2. NODES   = Functions that do work (receive state, return updates)
# 3. EDGES   = Connections between nodes (normal or conditional)
# 4. COMPILE = Turn the graph into a runnable application


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Simple Linear Graph (Research → Summarize)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("EXAMPLE 1: Simple linear graph")
print("=" * 55)

# STEP 1: Define State
class ResearchState(TypedDict):
    topic: str          # Input: what to research
    research: str       # Filled by research node
    summary: str        # Filled by summarize node

# STEP 2: Define Nodes (functions that process state)
def research_node(state: ResearchState) -> dict:
    """Research a topic using the LLM."""
    topic = state["topic"]
    response = model.invoke(f"Research this topic thoroughly in 3-4 sentences: {topic}")
    return {"research": response.content}
    # Returns a dict that UPDATES the state.
    # Only the keys you return get updated — "topic" stays unchanged.

def summarize_node(state: ResearchState) -> dict:
    """Summarize the research into bullet points."""
    research = state["research"]
    response = model.invoke(f"Summarize in 3 bullet points:\n{research}")
    return {"summary": response.content}

# STEP 3: Build the Graph
graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("research", research_node)
graph.add_node("summarize", summarize_node)

# Add edges (flow direction)
graph.add_edge(START, "research")           # Start → research
graph.add_edge("research", "summarize")     # research → summarize
graph.add_edge("summarize", END)            # summarize → end

# STEP 4: Compile
app = graph.compile()

# STEP 5: Run!
result = app.invoke({"topic": "Quantum Computing"})
print(f"Topic: {result['topic']}")
print(f"Research: {result['research'][:100]}...")
print(f"Summary:\n{result['summary']}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Three-Step Pipeline (Research → Fact Check → Report)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("EXAMPLE 2: Three-step pipeline")
print("=" * 55)

class PipelineState(TypedDict):
    topic: str
    research: str
    fact_check: str
    report: str

def do_research(state: PipelineState) -> dict:
    response = model.invoke(f"Research '{state['topic']}' in 2 sentences.")
    return {"research": response.content}

def do_fact_check(state: PipelineState) -> dict:
    response = model.invoke(
        f"Fact-check this research. Say 'VERIFIED' or note issues:\n{state['research']}"
    )
    return {"fact_check": response.content}

def write_report(state: PipelineState) -> dict:
    response = model.invoke(
        f"Write a brief report combining:\nResearch: {state['research']}\nFact check: {state['fact_check']}"
    )
    return {"report": response.content}

pipeline = StateGraph(PipelineState)
pipeline.add_node("research", do_research)
pipeline.add_node("fact_check", do_fact_check)
pipeline.add_node("write_report", write_report)

pipeline.add_edge(START, "research")
pipeline.add_edge("research", "fact_check")
pipeline.add_edge("fact_check", "write_report")
pipeline.add_edge("write_report", END)

app2 = pipeline.compile()
result = app2.invoke({"topic": "Python programming language"})
print(f"Report:\n{result['report']}")


# ── VISUALIZING GRAPHS ────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("BONUS: Visualize the graph")
print("=" * 55)

# Print as Mermaid diagram (can paste into mermaid.live)
print(app.get_graph().draw_mermaid())


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. StateGraph(TypedDict) — create a graph with typed state
# 2. graph.add_node("name", function) — add processing steps
# 3. graph.add_edge(START, "first") — connect nodes
# 4. graph.compile() — turn into runnable app
# 5. app.invoke(initial_state) — run the graph
# 6. Nodes receive state, return dict to UPDATE specific keys
# 7. Data flows: START → node1 → node2 → ... → END
#
# NEXT: step_02_conditional_edges.py — Branching and loops
