---
title: Agentic RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Agentic RAG with Multi-Agent Collaboration (CrewAI + LangGraph)

[![CrewAI](https://img.shields.io/badge/Framework-CrewAI-red)](https://crewai.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-green)](https://modelcontextprotocol.io)

A sophisticated Agentic RAG system that orchestrates a "Crew" of specialized AI agents to retrieve, summarize, architect, and audit information from local documents. This project implements a high-integrity workflow that separates content generation from technical auditing to ensure zero hallucinations.

## 🚀 Key Features

* **Multi-Agent Orchestration**: Managed via **LangGraph** for reliable state transitions and **CrewAI** for autonomous task execution.
* **Model Context Protocol (MCP)**: Utilizes the MCP Filesystem server to allow agents to persist technical summaries to the local disk, creating a "source of truth" for the audit phase.
* **Hybrid Multi-LLM Setup**: 
    * **Gemini 2.5 Flash**: Optimized for high-speed summarization and drafting.
    * **Llama 3.3 70B (via Groq)**: Acts as the "Technical Auditor" for rigorous quality control.
* **Production Reliability**: Implements **Sequential Process** and **Max RPM handling** to navigate Free Tier API limits while maintaining deep analysis.
* **Direct FAISS Integration**: Python-native vector search for sub-5s retrieval latency.

## 🏗 Workflow Architecture

1.  **Retrieve Node**: Direct FAISS search retrieves relevant document chunks.
2.  **Summarize Node**: `Data Summarizer` (Gemini) condenses info and saves it via **MCP**.
3.  **Generate Node**: `Response Architect` (Gemini) crafts the professional response.
4.  **Verify Node**: `Citation Manager` & `Technical Auditor` (Llama 70B) perform a final two-step audit for citations and accuracy.

## 📁 Project Structure
├── agents/          # CrewAI Agent definitions
├── workflow/        # LangGraph state machine & nodes
├── mcp_config/      # MCP tool & server configurations
├── rag/             # FAISS indexing & embedding logic
├── summaries/       # Local storage for AI-generated summaries (MCP managed)
└── app/             # FastAPI entry point

## 🛠 Setup & Installation

### Prerequisites
- Python 3.10+
- [Node.js](https://nodejs.org/) (Required for the MCP Filesystem server)

### 1. Installation

git clone https://github.com/mohamed-azaudeen/Agentic-RAG-with-Multi-Agent-Collaboration-CrewAI-LangGraph-.git
cd local_rag_mcp_bot
pip install -r requirements.txt

### 2. Configure Environment
Create a .env file in the root directory:

**GOOGLE_API_KEY=** your_gemini_api_key
**GROQ_API_KEY=** your_groq_api_key

### 3. Execution
You need to run three components simultaneously in separate terminals:

**- Terminal 1 (MCP Server):**
npx -y @modelcontextprotocol/server-filesystem "./summaries"

**- Terminal 2 (FastAPI Backend):**
uvicorn app.main:app --reload

**- Terminal 3 (Streamlit UI):**
streamlit run ui.py