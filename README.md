# 🧠 LangGraph AI Agent with Tool Calling

> A stateful, multi-turn AI agent built with LangGraph that reasons and uses tools to answer questions. Demonstrates production agent architecture with nodes, edges, conditional routing, and LangSmith observability.

Built by **Syed Muhammad Mehmam** — AI Engineer | [LinkedIn](https://linkedin.com/in/muhammad-mehmam) | [GitHub](https://github.com/Mehmaam99)

---

## 🎯 What This Does

An AI agent that can:
- 🔢 **Calculate** — math expressions via a safe calculator tool
- 🕐 **Tell time** — current date/time via datetime tool
- 📊 **Analyze text** — word count, sentence stats, top words
- 🔍 **Search knowledge base** — AI/ML technical questions
- 💬 **Multi-turn conversation** — remembers context across messages

## 🏗️ LangGraph Architecture

```
[START]
   │
   ▼
[agent node]  ←─────────────────────┐
   │                                │
   ▼ should_continue()              │
   ├── "tools" ──► [tool node] ─────┘  (loop until done)
   └── "end"   ──► [END]
```

**Key concepts demonstrated:**
- **Nodes**: `agent_node` (LLM reasoning) + `ToolNode` (tool execution)
- **Edges**: Conditional routing based on whether LLM called a tool
- **State**: `AgentState` TypedDict flows through every node explicitly
- **Loop**: Agent ↔ Tools loop until agent gives final answer

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Framework | LangGraph | Stateful, cyclical agent execution |
| LLM | Groq (Llama3-8b) | Free, fast inference |
| Tool Calling | LangChain Tools | Decorator-based tool definition |
| Observability | LangSmith (optional) | Trace every node, token, latency |
| API | FastAPI | Production-ready async API |

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/Mehmaam99/langgraph-agent
cd langgraph-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
export GROQ_API_KEY="your-groq-key"          # Required — free at console.groq.com
export LANGCHAIN_API_KEY="your-ls-key"        # Optional — enables LangSmith tracing

# 4. Run
uvicorn app.main:app --reload --port 8002

# 5. Open browser
# http://localhost:8002
```

## 💡 Real-World Application

This project demonstrates the **same agent architecture** I used to build a production chatbot for a startup client — where LangGraph's conditional routing handled: FAQ retrieval, general conversation, and fallback responses — all based on classified user intent. LangSmith monitored every run in production.

## 📁 Project Structure

```
project3_langgraph_agent/
├── app/
│   └── main.py         # LangGraph agent + FastAPI
├── static/
│   └── index.html      # Chat UI with tool indicators
├── requirements.txt
└── README.md
```

## 🔑 Why LangGraph > LangChain Chains

| Feature | LangChain Chain | LangGraph |
|---------|----------------|-----------|
| Flow | Linear A→B→C | Graph with loops |
| State | Implicit | Explicit TypedDict |
| Branching | ❌ No | ✅ Conditional edges |
| Loops | ❌ No | ✅ Agent can retry |
| Observability | Limited | Full LangSmith integration |
