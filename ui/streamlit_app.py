import streamlit as st
import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agentic RAG Bot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("⚙️ System Status")
    if st.button("Check Backend Health"):
        try:
            health = requests.get(f"{BASE_URL}/health").json()
            st.success(f"Backend: {health['status']}")
            st.info(f"Gemini API: {health['google_api']}")
            st.info(f"Groq API: {health['groq_api']}")
        except Exception as e:
            st.error(f"Backend Offline: {e}")
    
    st.divider()
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Add to Knowledge Base", type=["txt", "pdf", "docx", "csv"])
    
    if uploaded_file is not None:
        if st.button("🚀 Upload & Process"):
            with st.spinner("Uploading and indexing..."):
                try:
                    INTERNAL_URL = "http://127.0.0.1:8000/ingest"
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(INTERNAL_URL, files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        st.toast("Ingestion started!", icon="🚀")
                    else:
                        st.error(f"Upload Failed: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Internal Connection Error: {e}")

st.title("📲 Agentic RAG Explorer")
st.markdown("""
*Powered by **CrewAI** & **LangGraph***
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        with status_placeholder.status("🚀 Crew is collaborating...", expanded=True) as status:
            st.write("🔍 Searching FAISS vectorstore...")
            try:
                response = requests.post(
                    f"{BASE_URL}/ask", 
                    json={"query": query}, 
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    latency = data.get("latency")
                    
                    st.write(f"⚖️ Finalizing audit...")
                    status.update(label=f"✅ Complete ({latency})", state="complete", expanded=False)
                    
                    st.write(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                else:
                    status.update(label="❌ Backend Error", state="error")
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except requests.exceptions.Timeout:
                status.update(label="🕒 Timeout", state="error")
                st.error("The agents are taking a long time. This is normal for 70B models.")
            except Exception as e:
                status.update(label="⚠️ Connection Failed", state="error")
                st.error(f"Could not reach FastAPI: {e}")