# ============================================================
# LANGGRAPH — STEP 06: Checkpointing and Persistence
# Goal: Save graph state across runs. Resume after failure.
#       Multi-turn conversations with thread_id.
# Run:  python step_06_checkpointing.py
# ============================================================

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
# InMemorySaver — stores checkpoints in RAM (resets on restart)

# For persistent storage:
# from langgraph.checkpoint.sqlite import SqliteSaver
# → stores in a file, survives restarts

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: WHAT IS CHECKPOINTING? ──────────────────────────────────────────
#
# Checkpointing = saving the graph's state at each step.
#
# Benefits:
#   ✅ Resume from where you left off (after crash/restart)
#   ✅ Multi-turn conversations (remember previous messages)
#   ✅ Time-travel debugging (go back to any step)
#   ✅ Required for human-in-the-loop (step 04)
#   ✅ Multiple users with isolated sessions (thread_id)
#
# Without checkpointing:
#   Each invoke() is completely independent. No memory.
#
# With checkpointing:
#   Same thread_id = same conversation = remembers everything.


# ── BUILD A CHATBOT WITH MEMORY ───────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: ChatState) -> dict:
    system = SystemMessage(
        content="You are a helpful assistant. Remember everything the user tells you."
    )
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# KEY: Compile with checkpointer
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)


# ── PART 1: MULTI-TURN CONVERSATION ──────────────────────────────────────────

print("=" * 55)
print("PART 1: Multi-turn conversation (same thread)")
print("=" * 55)

# thread_id identifies a conversation:
config = {"configurable": {"thread_id": "bharath-chat"}}

result = app.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Bharath and I work at Infosys.")]},
    config=config,
)
print(f"Turn 1: {result['messages'][-1].content}")

result = app.invoke(
    {"messages": [HumanMessage(content="What company do I work at?")]},
    config=config,
)
print(f"Turn 2: {result['messages'][-1].content}")
# → "You work at Infosys!" — REMEMBERS because same thread_id!

result = app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config,
)
print(f"Turn 3: {result['messages'][-1].content}")
# → "Your name is Bharath!" — full conversation context available


# ── PART 2: ISOLATED SESSIONS (different thread_ids) ──────────────────────────

print("\n" + "=" * 55)
print("PART 2: Isolated sessions")
print("=" * 55)

# User A
config_a = {"configurable": {"thread_id": "alice"}}
app.invoke(
    {"messages": [HumanMessage(content="I'm Alice, I love Python.")]},
    config=config_a,
)

# User B
config_b = {"configurable": {"thread_id": "bob"}}
app.invoke(
    {"messages": [HumanMessage(content="I'm Bob, I prefer Rust.")]},
    config=config_b,
)

# They don't mix!
result_a = app.invoke(
    {"messages": [HumanMessage(content="What's my name and favorite language?")]},
    config=config_a,
)
result_b = app.invoke(
    {"messages": [HumanMessage(content="What's my name and favorite language?")]},
    config=config_b,
)
print(f"Alice's session: {result_a['messages'][-1].content}")
print(f"Bob's session: {result_b['messages'][-1].content}")


# ── PART 3: VIEWING STATE ────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Inspect current state")
print("=" * 55)

state = app.get_state(config)
print(f"Messages in Bharath's session: {len(state.values['messages'])}")
print(f"Next node: {state.next}")  # Empty if graph is done

for msg in state.values["messages"]:
    role = type(msg).__name__
    print(f"  [{role}] {msg.content[:60]}...")


# ── PART 4: STATE HISTORY (Time Travel) ───────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: State history (all checkpoints)")
print("=" * 55)

for i, snapshot in enumerate(app.get_state_history(config)):
    msg_count = len(snapshot.values.get("messages", []))
    print(f"  Checkpoint {i}: {msg_count} messages")
# You can jump back to any checkpoint!


# ── PART 5: SQLITE PERSISTENCE (survives restarts) ────────────────────────────

print("\n" + "=" * 55)
print("PART 5: SQLite persistence (saves to file)")
print("=" * 55)

print("""
# For production — save to disk:

from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as db:
    app = graph.compile(checkpointer=db)
    
    # Now conversations persist across program restarts!
    result = app.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "user-123"}}
    )

# Other persistent checkpointers:
#   PostgresSaver  — production database
#   RedisSaver     — fast, distributed
""")


# ── PART 6: STREAMING WITH CHECKPOINTING ─────────────────────────────────────

print("=" * 55)
print("PART 6: Streaming with checkpointing")
print("=" * 55)

config_stream = {"configurable": {"thread_id": "stream-demo"}}

print("Streaming response: ", end="", flush=True)
for chunk in app.stream(
    {"messages": [HumanMessage(content="Write a haiku about programming")]},
    config=config_stream,
    stream_mode="values",
):
    latest = chunk["messages"][-1]
    if type(latest).__name__ == "AIMessage":
        print(latest.content)

# The streamed conversation is ALSO checkpointed!
# Next invoke with same thread_id will have this context.


# ── CHECKPOINTER COMPARISON ───────────────────────────────────────────────────
#
# | Checkpointer     | Storage     | Survives restart? | Use case           |
# |------------------|-------------|-------------------|--------------------|
# | InMemorySaver    | RAM         | No                | Development/testing|
# | SqliteSaver      | SQLite file | Yes               | Single-server prod |
# | PostgresSaver    | PostgreSQL  | Yes               | Multi-server prod  |
# | RedisSaver       | Redis       | Yes (TTL)         | Fast, distributed  |


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. checkpointer=InMemorySaver() — enable state persistence
# 2. thread_id in config — identifies a conversation session
# 3. Same thread_id = continues conversation (remembers)
# 4. Different thread_id = isolated session (separate memory)
# 5. app.get_state(config) — inspect current state
# 6. app.get_state_history(config) — all checkpoints (time travel)
# 7. SqliteSaver — persists across restarts (production use)
# 8. Streaming works with checkpointing — state is saved after each node
#
# ═══════════════════════════════════════════════════════════════════════════════
# END OF LANGGRAPH CODES — All 6 steps cover:
#   01: StateGraph Basics (State, Nodes, Edges, START/END)
#   02: Conditional Edges and Loops (routing, cycles, retry)
#   03: Agent with Tools (ReAct loop, ToolNode, add_messages)
#   04: Human-in-the-Loop (interrupt, approve/reject)
#   05: Multi-Agent (Supervisor pattern, specialist tools)
#   06: Checkpointing (persistence, thread_id, state history)
# ═══════════════════════════════════════════════════════════════════════════════

This