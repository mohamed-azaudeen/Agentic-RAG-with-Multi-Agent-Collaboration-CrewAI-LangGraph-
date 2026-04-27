import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Bot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("⚙️ System Status")
    if st.button("Check Backend Health"):
        try:
            health = requests.get(f"{API_URL}/health").json()
            st.success(f"Backend: {health['status']}")
            st.info(f"Gemini API: {health['google_api']}")
            st.info(f"Groq API: {health['groq_api']}")
        except:
            st.error("Backend Offline")
    
    st.divider()
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Add to Knowledge Base", type=["txt", "pdf", "docx", "csv"])
    
    if uploaded_file is not None:
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name}")

        with st.spinner("Updating Vectorstore..."):
            response = requests.post(f"{API_URL}/ingest", json={"filename": uploaded_file.name})
            if response.status_code == 200:
                st.toast("Ingestion started in background!", icon="✅")
            else:
                st.error("Ingestion failed.")



st.title("🤖 Agentic RAG Explorer")
st.markdown("""
*Powered by **CrewAI** (Summarizer, Architect, Citation, Critic) & **MCP***
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask about your documents (e.g., 'What is the remote work policy?')"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        with status_placeholder.status("🚀 Crew is collaborating...", expanded=True) as status:
            st.write("🔍 Searching FAISS vectorstore...")
            # We use a POST request to our FastAPI /ask endpoint
            try:
                response = requests.post(f"{API_URL}/ask", json={"query": query}, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    latency = data.get("latency")
                    
                    st.write(f"📝 Summarizing context with Gemini Flash...")
                    st.write(f"⚖️ Verifying citations and auditing with Llama 70B...")
                    status.update(label=f"✅ Complete ({latency})", state="complete", expanded=False)
                    
                    st.write(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                else:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Backend Error: {response.text}")
            
            except requests.exceptions.Timeout:
                status.update(label="🕒 Timeout", state="error")
                st.error("The request took too long. Check if your MCP servers are running.")
            except Exception as e:
                status.update(label="⚠️ Connection Failed", state="error")
                st.error(f"Could not reach FastAPI: {e}")