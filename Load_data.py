import os
from dotenv import load_dotenv
from langchain.text_splitter import RecurarsiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader


load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HuggingFace_API_KEY")

load_docs():
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            return docs
        
        