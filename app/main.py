import os
import logging
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from workflow.graph import workflow
from ingestion.loader import load_documents
from rag.vectorstore import create_vector_store

load_dotenv()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

chat_history = []
vectorstore = None

class Question(BaseModel):
    query: str
    session_id: Optional[str] = "default_user"

@app.on_event("startup")
def startup_event():
    global vectorstore
    logging.info("🚀 System starting up... initializing folders and vectorstore.")

    # Ensure directories exist for persistent storage
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if os.path.exists("vectorstore/index.faiss"):
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ Existing FAISS index loaded successfully.")
        else:
            logging.info("ℹ️ No existing index found. Checking 'data' folder...")
            docs = load_documents("data")
            if docs:
                vectorstore = create_vector_store(docs, embeddings)
                logging.info("✅ New FAISS index created.")
            else:
                logging.warning("⚠️ 'data' folder is empty. Upload a file via the UI to begin.")
    except Exception as e:
        logging.error(f"❌ Startup Error: {str(e)}")

@app.post("/ask")
async def ask_question(q: Question):
    global chat_history
    if not q.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        start_time = time.time()
        logging.info(f"📩 Processing query: {q.query}")

        loop = asyncio.get_event_loop()
        inputs = {"query": q.query, "history": chat_history}
        
        # Execute the LangGraph workflow in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None, 
            lambda: workflow.invoke(inputs)
        )
        
        elapsed = time.time() - start_time
        chat_history = result.get("history", chat_history)
        answer = result.get("final_output", "The agents could not produce a verified answer.")
        
        logging.info(f"✅ Workflow complete in {elapsed:.2f}s")
        
        return {
            "answer": answer,
            "latency": f"{elapsed:.2f}s",
            "docs_retrieved": len(result.get("raw_docs", []))
        }

    except Exception as e:
        logging.error(f"❌ Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Handles file upload and triggers background ingestion.
    """
    try:
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logging.info(f"📂 File '{file.filename}' uploaded successfully.")


        def sync_ingestion():
            global vectorstore
            logging.info("🔄 Background re-ingestion starting...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docs = load_documents("data")
            if docs:
                vectorstore = create_vector_store(docs, embeddings)
                logging.info("✅ Vectorstore update complete.")
            else:
                logging.warning("⚠️ Ingestion skipped: No valid documents found.")


        background_tasks.add_task(sync_ingestion)
        
        return {"status": "success", "message": f"'{file.filename}' uploaded. Processing in background."}

    except Exception as e:
        logging.error(f"❌ Upload/Ingest Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Returns the current state of the backend."""
    return {
        "status": "ready" if vectorstore else "initializing",
        "vectorstore_path": "exists" if os.path.exists("vectorstore/index.faiss") else "missing",
        "google_api": "configured" if os.getenv("GOOGLE_API_KEY") else "missing",
        "groq_api": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "data_count": len(os.listdir("data")) if os.path.exists("data") else 0
    }