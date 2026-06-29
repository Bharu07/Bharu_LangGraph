# ============================================================
# LANGGRAPH — STEP 04: Human-in-the-Loop
# Goal: Pause the graph before sensitive actions, let a human
#       approve or reject, then resume execution.
# Run:  python step_04_human_in_the_loop.py
# ============================================================

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
# InMemorySaver = stores checkpoints in RAM (required for interrupts)

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: WHY HUMAN-IN-THE-LOOP? ──────────────────────────────────────────
#
# Some actions are DANGEROUS if the agent does them automatically:
#   • Sending emails to customers
#   • Deleting data from databases
#   • Making payments or refunds
#   • Publishing content publicly
#
# Solution: PAUSE the graph before the dangerous step.
# A human reviews what the agent wants to do, then approves/rejects.
#
# LangGraph supports this with:
#   interrupt_before=["node_name"]  → pause BEFORE this node runs
#   interrupt_after=["node_name"]   → pause AFTER this node runs


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
def search_info(query: str) -> str:
    """Search for information (safe action)."""
    return f"Found info about '{query}': It's a widely discussed topic."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone (DANGEROUS — needs approval)."""
    # In production, this would actually send an email!
    return f"Email sent to {to} with subject '{subject}'"

@tool
def delete_record(record_id: str) -> str:
    """Delete a record from the database (DANGEROUS — needs approval)."""
    return f"Record {record_id} deleted successfully."

tools = [search_info, send_email, delete_record]


# ── BUILD AGENT WITH INTERRUPT ────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState) -> dict:
    system = SystemMessage(content="You are helpful. Use tools when needed.")
    response = model_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "model")

# KEY: Compile with checkpointer AND interrupt_before
memory = InMemorySaver()
agent = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"],  # PAUSE before tool execution!
)
# The graph will STOP just before the "tools" node runs.
# This gives a human time to review what tool calls the agent wants to make.


# ── EXAMPLE 1: SAFE ACTION (no interrupt needed in practice) ──────────────────

print("=" * 55)
print("EXAMPLE 1: Safe action (search)")
print("=" * 55)

config = {"configurable": {"thread_id": "session-1"}}

# First invoke — agent will request to call search_info
result = agent.invoke(
    {"messages": [HumanMessage(content="Search for info about Python")]},
    config=config,
)

# Graph paused before tools! Let's see what it wants to do:
state = agent.get_state(config)
last_msg = state.values["messages"][-1]
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print(f"Agent wants to call: {[tc['name'] for tc in last_msg.tool_calls]}")
    print(f"With args: {[tc['args'] for tc in last_msg.tool_calls]}")
    print("→ APPROVED (safe action)")

# Resume execution (approve by calling invoke with None):
result = agent.invoke(None, config=config)
print(f"Final answer: {result['messages'][-1].content}")


# ── EXAMPLE 2: DANGEROUS ACTION (human rejects) ──────────────────────────────

print("\n" + "=" * 55)
print("EXAMPLE 2: Dangerous action (email — REJECTED)")
print("=" * 55)

config2 = {"configurable": {"thread_id": "session-2"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Send an email to boss@company.com saying I quit")]},
    config=config2,
)

# Graph paused! Human reviews:
state = agent.get_state(config2)
last_msg = state.values["messages"][-1]
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print(f"Agent wants to call: {[tc['name'] for tc in last_msg.tool_calls]}")
    print(f"With args: {[tc['args'] for tc in last_msg.tool_calls]}")
    print("→ REJECTED! (too dangerous)")

# Reject by updating state (remove tool calls, add rejection message):
from langchain.messages import AIMessage
agent.update_state(
    config2,
    {"messages": [AIMessage(content="I cannot send that email. It was rejected by the human reviewer.")]},
)

# Resume:
result = agent.invoke(None, config=config2)
print(f"Final answer: {result['messages'][-1].content}")


# ── EXAMPLE 3: VIEWING STATE HISTORY ─────────────────────────────────────────

print("\n" + "=" * 55)
print("EXAMPLE 3: State history (time travel)")
print("=" * 55)

# View all checkpoints for a session:
for state in agent.get_state_history(config):
    msgs = state.values.get("messages", [])
    print(f"  Checkpoint: {len(msgs)} messages | Next: {state.next}")


# ── USE CASES FOR HUMAN-IN-THE-LOOP ──────────────────────────────────────────
#
# • Approve before sending emails/notifications
# • Review before database modifications (delete, update)
# • Confirm before making payments or refunds
# • Quality check before publishing content
# • Provide additional context when agent is stuck
# • Override agent's decision with human judgment


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. interrupt_before=["node"] — pause BEFORE that node runs
# 2. Requires checkpointer (InMemorySaver) to save state at pause
# 3. agent.get_state(config) — see what the agent wants to do
# 4. agent.invoke(None, config) — APPROVE and resume
# 5. agent.update_state(config, values) — REJECT/modify and resume
# 6. get_state_history() — view all checkpoints (time travel)
# 7. thread_id in config — identifies the session
#
# NEXT: step_05_multi_agent.py — Multiple agents collaborating
