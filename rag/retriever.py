from langchain_community.vectorstores import FAISS

def load_retriever(embeddings, path="vectorstore"):
    db = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 3})