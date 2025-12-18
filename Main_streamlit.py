import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.agents import create_agent


load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HuggingFace_API_KEY")

st.write("EHR Q&A Agent Challenge")
question = st.text_input("Enter your question about the EHR documents:")

@st.cache_resource
def setup_rag():
    documents = load_docs()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    agent = create_agent(llm=HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":512}, huggingfacehub_api_token=HUGGINGFACE_API_KEY),
    system_prompt="Is there any history of illness for the patient White, Alan Joseph?",)

    agent.invoke(
    {"messages": [{"role": "user", "content": "What is the WBC and RBC count of this patient?"}]})
    
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":512}, huggingfacehub_api_token=HUGGINGFACE_API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

if question:
    qa_chain = setup_rag()
    response = qa_chain.run(question)
    st.write(response)

