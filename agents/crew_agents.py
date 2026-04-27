import os
from crewai import Agent, LLM
from mcp_config.tools import MCP_CONFIG

gemini_flash = LLM(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

groq_critic_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

summarizer_agent = Agent(
    role="Data Summarizer",
    goal="Distill large document chunks into clear, bulleted insights.",
    backstory="You specialize in technical summarization and data extraction.",
    llm=gemini_flash,
    max_rpm=2,
    mcps=[MCP_CONFIG["filesystem"]],
    cache=True,
    verbose=True
)

answer_agent = Agent(
    role="Response Architect",
    goal="Construct a final user-friendly answer based on summaries.",
    backstory="You are a professional technical writer skilled in synthesis.",
    llm=gemini_flash,
    max_rpm=2,
    cache=True,
    verbose=True
)

citation_agent = Agent(
    role="Citation Manager",
    goal="Fetch metadata and ensure all claims have proper sources.",
    backstory="You maintain academic integrity using specialized research tools.",
    llm=groq_critic_llm,
    max_rpm=2,
    cache=True,
    verbose=True
)

critic_agent = Agent(
    role="Technical Auditor",
    goal="verify the final answer for technical accuracy and hallucinations.",
    backstory="A senior-level critic who flags inconsistencies or false claims.",
    llm=groq_critic_llm,
    cache=True,
    verbose=True
)