"""
========================================================
PROJECT 3: LangGraph AI Agent with Tool Calling
========================================================
Author      : Syed Muhammad Mehmam
Tech Stack  : FastAPI + LangGraph + LangChain + Groq + LangSmith
Description : A production-ready AI agent built with LangGraph
              that can reason, use tools (web search, calculator,
              text summarizer), and maintain conversation state
              across multi-turn interactions.

              This demonstrates the same agent architecture used
              in the startup chatbot project — with nodes, edges,
              conditional routing, and full LangSmith observability.
========================================================
"""

import os
import math
import json
from typing import TypedDict, Annotated, Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()  # Add this at the top

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")   # Optional: LangSmith tracing

# Enable LangSmith tracing if API key provided
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Agent-Portfolio"
    print("[INFO] LangSmith tracing enabled")

# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state that flows through every node in the LangGraph.

    This is the KEY difference from LangChain chains:
    - In a chain: state is implicit and linear
    - In LangGraph: state is EXPLICIT and passed between every node
    - Any node can read/write to state
    - The graph routes based on state values
    """
    messages: Annotated[list, add_messages]   # Full conversation history
    tool_calls_made: int                       # Track how many tools were called
    final_answer: str                          # The agent's final response

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this for any numerical calculations.
    Examples: '2 + 2', '15 * 8', 'sqrt(144)', '2 ** 10'
    """
    try:
        # Safe evaluation — only allows math operations
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """
    Get the current date and time.
    Use this when the user asks about the current time, date, or day.
    """
    now = datetime.utcnow()
    return f"Current UTC datetime: {now.strftime('%A, %B %d, %Y at %H:%M:%S UTC')}"


@tool
def text_analyzer(text: str) -> str:
    """
    Analyze a piece of text and return statistics.
    Use this when the user wants to analyze, summarize stats about, or understand text properties.
    Returns: word count, sentence count, avg word length, most common words.
    """
    words = text.lower().split()
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    if not words:
        return "No text provided"

    # Word frequency
    word_freq = {}
    stop_words = {'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    for word in words:
        clean = word.strip('.,!?;:')
        if clean and clean not in stop_words:
            word_freq[clean] = word_freq.get(clean, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    avg_len = sum(len(w) for w in words) / len(words)

    return (
        f"Text Analysis:\n"
        f"- Word count: {len(words)}\n"
        f"- Sentence count: {len(sentences)}\n"
        f"- Avg word length: {avg_len:.1f} characters\n"
        f"- Top words: {', '.join([f'{w}({c})' for w, c in top_words])}"
    )


@tool
def knowledge_base_search(query: str) -> str:
    """
    Search a simulated knowledge base about AI/ML topics.
    Use this when the user asks about AI, machine learning, LLMs, or related technical topics.
    """
    # Simulated KB — in production this would hit a real vector DB
    knowledge = {
        "rag": "RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. Documents are chunked, embedded, and stored in a vector DB. At query time, relevant chunks are retrieved and injected into the LLM prompt.",
        "langgraph": "LangGraph is a framework for building stateful, graph-based AI agents. Unlike linear LangChain chains, LangGraph uses nodes (functions) connected by edges (transitions) with explicit state management.",
        "yolo": "YOLO (You Only Look Once) is a real-time object detection algorithm. YOLOv8 processes images in a single forward pass through a CNN, detecting objects and drawing bounding boxes in milliseconds.",
        "embeddings": "Embeddings are dense vector representations of text. Similar text has similar vectors. Used in RAG to find relevant documents via cosine similarity search in vector databases like FAISS or ChromaDB.",
        "llm": "Large Language Models (LLMs) are transformer-based models trained on massive text datasets. They generate text by predicting the next token. Examples: GPT-4, Llama3, Claude.",
        "vector database": "Vector databases store and query high-dimensional embedding vectors. They enable fast semantic similarity search. Examples: FAISS (in-memory), ChromaDB (local), Pinecone (cloud).",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Found in knowledge base:\n{value}"

    return f"No specific entry found for '{query}'. Try asking about: RAG, LangGraph, YOLO, embeddings, LLMs, or vector databases."


# Register all tools
TOOLS = [calculator, get_current_datetime, text_analyzer, knowledge_base_search]

# ── LLM Setup ─────────────────────────────────────────────────────────────────

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=2048
)

# Bind tools to LLM — enables function/tool calling
llm_with_tools = llm.bind_tools(TOOLS)

# ── Graph Nodes ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools:
- calculator: for math calculations
- get_current_datetime: for current time/date
- text_analyzer: for analyzing text statistics
- knowledge_base_search: for AI/ML technical questions

INSTRUCTIONS:
- Think step by step before responding
- Use tools when they would give a better answer than your training knowledge
- Be concise and accurate
- If you use a tool, explain what you found
"""


def agent_node(state: AgentState) -> AgentState:
    """
    AGENT NODE — The brain of the agent.

    This node:
    1. Receives the current state (full conversation history)
    2. Sends messages to the LLM with tools available
    3. LLM decides: respond directly OR call a tool
    4. Returns updated state with LLM's response

    This is equivalent to the "reasoning" step in the agent loop.
    """
    messages = state["messages"]

    # Prepend system message if this is the first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Call LLM with tools
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "tool_calls_made": state.get("tool_calls_made", 0),
        "final_answer": ""
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    ROUTING FUNCTION — This is what makes LangGraph powerful.

    After the agent node runs, we check:
    - Did the LLM request a tool call? → route to "tools" node
    - Did the LLM give a direct answer? → route to "end"

    This conditional routing is impossible in a linear LangChain chain.
    """
    last_message = state["messages"][-1]

    # If LLM called tools, route to tool execution node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return "end"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Construct the LangGraph agent graph.

    Graph structure:
        [START]
           │
           ▼
       [agent]  ←─────────────────┐
           │                      │
           ▼ (conditional)        │
      ┌────┴─────┐                │
      │          │                │
    "tools"    "end"              │
      │          │                │
      ▼          ▼                │
    [tools]    [END]              │
      │                          │
      └──────────────────────────┘
         (loop back to agent)

    The agent loops until it decides to give a final answer (no more tool calls).
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))   # Built-in LangGraph tool executor

    # Set entry point
    graph.set_entry_point("agent")

    # Conditional edge from agent: tools or end
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )

    # After tools run, always go back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compile the graph once at startup
agent_graph = build_agent_graph()
print("[INFO] LangGraph agent compiled successfully")

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LangGraph AI Agent API",
    description="Multi-tool AI agent built with LangGraph — demonstrates stateful agent with tool calling",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory conversation store (use Redis/DB in production)
conversations: dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    tools_used: list[str]
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the LangGraph agent.
    The agent will reason, use tools if needed, and return an answer.
    Conversation history is maintained per session_id.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Get or create conversation history
    session_messages = conversations.get(request.session_id, [])
    session_messages.append(HumanMessage(content=request.message))

    try:
        # Run the LangGraph agent
        result = agent_graph.invoke({
            "messages": session_messages,
            "tool_calls_made": 0,
            "final_answer": ""
        })

        # Extract final response
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)

        # Extract which tools were used
        tools_used = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                tools_used.append(msg.name if hasattr(msg, 'name') else "tool")

        # Update conversation history
        conversations[request.session_id] = result["messages"]

        return ChatResponse(
            response=response_text,
            tools_used=list(set(tools_used)),
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    conversations.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/tools")
async def list_tools():
    """List all available agent tools"""
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in TOOLS
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "tools_available": len(TOOLS)}
