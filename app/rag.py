import pandas as pd
#from langchain.schema import Document
from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "data/v2-carapi-datafeed-sample.xlsx"
VECTOR_PATH = "vectorstore/faiss_index"

def load_excel_as_documents():
    xls = pd.ExcelFile(DATA_PATH)
    documents = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df.dropna(how="all")

        for _, row in df.iterrows():
            text = f"Sheet: {sheet}\n"
            for col, val in row.items():
                text += f"{col}: {val}\n"

            documents.append(
                Document(
                    page_content=text,
                    metadata={"sheet": sheet}
                )
            )

    return documents


def build_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("HUGGINGFACE_MODEL")
    )

    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(VECTOR_PATH, embeddings)

    docs = load_excel_as_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTOR_PATH)

    return vectorstore