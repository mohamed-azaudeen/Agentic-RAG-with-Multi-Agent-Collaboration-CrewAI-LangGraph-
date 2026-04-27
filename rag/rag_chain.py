from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def generate_answer(query, docs):
    context = "\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Use context below to answer the question.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {query}
        """
    )

    chain = prompt_template | llm

    response = chain.invoke({"context": context, "query": query})
    return response.content