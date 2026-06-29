# LangGraph — Step-by-Step Guide

A hands-on tutorial series for learning LangGraph from beginner to
advanced graph-based agent orchestration using Azure OpenAI as the
LLM provider.

## What's Covered

| Step | Topic |
|------|-------|
| 01 | State Graph Basics |
| 02 | Conditional Edges |
| 03 | Agent with Tools |
| 04 | Human-in-the-Loop |
| 05 | Multi-Agent Systems |
| 06 | Checkpointing |

Learn concepts including:

- State graphs
- Nodes and edges
- State management
- Conditional routing
- Tool-calling agents
- Human-in-the-loop workflows
- Multi-agent systems
- Graph orchestration
- Checkpointing
- Persistent execution
- Interrupt and resume
- Production workflow patterns

---

## Setup

1. Clone this repository.
2. Create a virtual environment:
    python -m venv .venv

3. Install dependencies:
    pip install -r requirements.txt

4. Create a .env file:
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
    AZURE_OPENAI_API_KEY=your-api-key
    AZURE_OPENAI_DEPLOYMENT=your-deployment-name
    OPENAI_API_VERSION=2025-03-01-preview

5. Run any example:
    python step_01_state_graph_basics.py

## Tech Stack
    Python 3.11+
    LangGraph
    LangChain v1
    langchain-openai
    Azure OpenAI (GPT-4o)

## GitHub Topics
    langgraph
    langchain
    azure-openai
    python
    ai-agents
    multi-agent
    graph
    state-machine
    workflow
    generative-ai




