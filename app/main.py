from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.rag import build_or_load_vectorstore
from app.llm import load_llm

app = FastAPI(title="Car Data Chatbot")

# Load components
vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = load_llm()

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Answer the question ONLY using the context below.
    If the answer is not present in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
)

# -------- RAG PIPELINE (LangChain 1.x) --------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    # Get retrieved docs first (for fallback logic)
    docs = retriever.get_relevant_documents(request.question)

    # If no relevant data → fallback to pure LLM
    if not docs:
        answer = llm.invoke(request.question)
        return {
            "answer": answer.content,
            "source": "LLM (No matching data found)"
        }

    # RAG answer
    response = rag_chain.invoke(request.question)

    return {
        "answer": response.content,
        "source": "Excel Data"
    }
