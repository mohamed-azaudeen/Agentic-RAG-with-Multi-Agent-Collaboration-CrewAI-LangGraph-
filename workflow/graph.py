import time
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from crewai import Task, Crew, Process
from mcp_config.tools import direct_vector_search
from agents.crew_agents import summarizer_agent, answer_agent, citation_agent, critic_agent


class WorkFlowState(TypedDict):
    query: str
    raw_docs: List[dict]
    summary: str
    base_answer: str
    final_output: str
    log: str
    history: List[dict]


def extract_output(result):
    if result is None: return ""
    return result.raw if hasattr(result, "raw") else str(result)


def retrieve_node(state: WorkFlowState) -> dict:
    """Direct FAISS search - Bypasses LLM for speed."""
    query = state["query"]
    print(f"\n🔍 [1/4] Starting Direct Retrieval for: {query}")
    
    start_time = time.time()
    try:
        docs = direct_vector_search(query)
        elapsed = time.time() - start_time
        
        print(f"⏱ Retrieval took {elapsed:.2f}s | Found {len(docs)} chunks.")
        return {"raw_docs": docs, "log": f"Successfully retrieved {len(docs)} documents."}
    except Exception as e:
        print(f"❌ Retrieval Error: {str(e)}")
        return {"raw_docs": [], "log": f"Error: {str(e)}"}

def summarize_node(state: WorkFlowState) -> dict:
    """Agent 1: Condenses the raw chunks into key insights."""
    if not state["raw_docs"]:
        return {"summary": "No context available."}
    
    print("📝 [2/4] Summarizing documents...")
    time.sleep(12)
    context = "\n".join([d['text'] for d in state["raw_docs"]])
    
    task = Task(
        description=f"Synthesize these document chunks into key bullet points relevant to: {state['query']}\n\nContext: {context}",
        expected_output="A structured summary of facts extracted from the documents.",
        agent=summarizer_agent
    )
    
    result = Crew(agents=[summarizer_agent], tasks=[task]).kickoff()
    return {"summary": extract_output(result)}

def generate_node(state: WorkFlowState) -> dict:
    """Agent 2: Creates the conversational response."""
    print("💡 [3/4] Generating initial answer...")
    time.sleep(12)

    task = Task(
        description=f"Using this summary: {state['summary']}, answer the user's question: {state['query']}",
        expected_output="A helpful, professional, and clear response.",
        agent=answer_agent
    )
    
    result = Crew(agents=[answer_agent], tasks=[task]).kickoff()
    return {"base_answer": extract_output(result)}

def verify_and_cite_node(state: WorkFlowState) -> dict:
    """Agents 3 & 4: Adds citations (Exa MCP) and audits for errors (Llama 70B)."""
    print("⚖️ [4/4] Running Citation and Quality Audit...")
    time.sleep(12)
    
    cite_task = Task(
        description=f"Add formal citations to this answer: {state['base_answer']}. Use your research tools to verify source names if needed.",
        expected_output="The answer with source references included.",
        agent=citation_agent
    )
    
    critic_task = Task(
        description=f"Review the cited answer for any hallucinations or inaccuracies based on the original summary: {state['summary']}.",
        expected_output="The final, polished, and verified answer.",
        agent=critic_agent
    )

    verification_crew = Crew(
        agents=[citation_agent, critic_agent], 
        tasks=[cite_task, critic_task],
        cache=True,
        process=Process.sequential,
        verbose=True
    )
    
    result = verification_crew.kickoff()
    
    new_history = state.get("history", [])
    new_history.append({"role": "user", "content": state["query"]})
    new_history.append({"role": "assistant", "content": extract_output(result)})
    
    return {"final_output": extract_output(result), "history": new_history}

workflow_graph = StateGraph(WorkFlowState)

workflow_graph.add_node("retrieve", retrieve_node)
workflow_graph.add_node("summarize", summarize_node)
workflow_graph.add_node("generate", generate_node)
workflow_graph.add_node("verify", verify_and_cite_node)

workflow_graph.add_edge(START, "retrieve")
workflow_graph.add_edge("retrieve", "summarize")
workflow_graph.add_edge("summarize", "generate")
workflow_graph.add_edge("generate", "verify")
workflow_graph.add_edge("verify", END)

workflow = workflow_graph.compile()