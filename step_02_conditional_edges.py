# ============================================================
# LANGGRAPH — STEP 02: Conditional Edges and Loops
# Goal: Route to different nodes based on state (if/else).
#       Create cycles (loops) for retry/iteration patterns.
# Run:  python step_02_conditional_edges.py
# ============================================================

from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: CONDITIONAL ROUTING (Quality Check)
#
#   research → quality_check → [score >= 7?]
#                                 YES → write_report → END
#                                 NO  → research (LOOP back!)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("EXAMPLE 1: Conditional routing with quality loop")
print("=" * 55)

class QualityState(TypedDict):
    topic: str
    research: str
    quality_score: int
    attempt: int
    report: str

def research(state: QualityState) -> dict:
    attempt = state.get("attempt", 0) + 1
    prompt = f"Research '{state['topic']}' thoroughly in 3-4 sentences."
    if attempt > 1:
        prompt += " Provide MORE detail and specific facts this time."
    response = model.invoke(prompt)
    print(f"  [Attempt {attempt}] Researching...")
    return {"research": response.content, "attempt": attempt}

def quality_check(state: QualityState) -> dict:
    response = model.invoke(
        f"Rate this research quality from 1-10 (just the number):\n{state['research']}"
    )
    try:
        score = int(response.content.strip().split()[0])
    except ValueError:
        score = 5
    print(f"  [Quality] Score: {score}/10")
    return {"quality_score": score}

def write_report(state: QualityState) -> dict:
    response = model.invoke(f"Write a brief report from:\n{state['research']}")
    return {"report": response.content}

# THE ROUTING FUNCTION — decides where to go next:
def decide_next(state: QualityState) -> str:
    """Route based on quality score."""
    if state["quality_score"] >= 7:
        return "write_report"    # Good enough → proceed
    if state.get("attempt", 0) >= 3:
        return "write_report"    # Max retries → proceed anyway
    return "research"            # Not good enough → LOOP BACK

# BUILD THE GRAPH:
graph = StateGraph(QualityState)
graph.add_node("research", research)
graph.add_node("quality_check", quality_check)
graph.add_node("write_report", write_report)

graph.add_edge(START, "research")
graph.add_edge("research", "quality_check")

# CONDITIONAL EDGE — the key concept!
graph.add_conditional_edges(
    "quality_check",     # FROM this node
    decide_next,         # USE this function to decide
    {
        "write_report": "write_report",  # If returns "write_report" → go here
        "research": "research",          # If returns "research" → LOOP BACK
    }
)

graph.add_edge("write_report", END)

app = graph.compile()
result = app.invoke({"topic": "LangGraph framework", "attempt": 0, "quality_score": 0})
print(f"\nFinal Report:\n{result['report'][:200]}...")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: MULTI-PATH ROUTING (Classify → Route to Specialist)
#
#   classify → [intent?]
#               "question"  → answer_node → END
#               "complaint" → complaint_node → END
#               "feedback"  → feedback_node → END
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("EXAMPLE 2: Multi-path routing")
print("=" * 55)

class RouterState(TypedDict):
    message: str
    intent: str
    response: str

def classify(state: RouterState) -> dict:
    response = model.invoke(
        f"Classify this message as 'question', 'complaint', or 'feedback'. "
        f"Reply with ONLY one word:\n{state['message']}"
    )
    intent = response.content.strip().lower()
    print(f"  [Classified as: {intent}]")
    return {"intent": intent}

def handle_question(state: RouterState) -> dict:
    response = model.invoke(f"Answer this question helpfully:\n{state['message']}")
    return {"response": f"[Q&A] {response.content}"}

def handle_complaint(state: RouterState) -> dict:
    response = model.invoke(
        f"Acknowledge this complaint empathetically and offer a solution:\n{state['message']}"
    )
    return {"response": f"[COMPLAINT] {response.content}"}

def handle_feedback(state: RouterState) -> dict:
    return {"response": f"[FEEDBACK] Thank you for your feedback! We appreciate it."}

def route_by_intent(state: RouterState) -> str:
    """Route to different handlers based on intent."""
    intent = state["intent"]
    if "question" in intent:
        return "handle_question"
    elif "complaint" in intent:
        return "handle_complaint"
    else:
        return "handle_feedback"

# Build multi-path graph:
router = StateGraph(RouterState)
router.add_node("classify", classify)
router.add_node("handle_question", handle_question)
router.add_node("handle_complaint", handle_complaint)
router.add_node("handle_feedback", handle_feedback)

router.add_edge(START, "classify")
router.add_conditional_edges("classify", route_by_intent, {
    "handle_question": "handle_question",
    "handle_complaint": "handle_complaint",
    "handle_feedback": "handle_feedback",
})
router.add_edge("handle_question", END)
router.add_edge("handle_complaint", END)
router.add_edge("handle_feedback", END)

app2 = router.compile()

# Test different messages:
messages = [
    "How do I reset my password?",
    "Your service is terrible! I've been waiting 3 hours!",
    "I love the new dark mode feature, great job!",
]

for msg in messages:
    result = app2.invoke({"message": msg})
    print(f"\n  Input: {msg}")
    print(f"  Response: {result['response'][:100]}...")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. add_conditional_edges(node, routing_fn, mapping) — branch based on state
# 2. Routing function: receives state → returns string (next node name)
# 3. Mapping dict: maps return values to node names
# 4. LOOPS: routing function can return a previous node → creates a cycle
# 5. Always add a max retry check to prevent infinite loops
# 6. Multi-path: one input, multiple possible paths through the graph
#
# NEXT: step_03_agent_with_tools.py — Build a tool-calling agent graph
