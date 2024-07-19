import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
load_dotenv()

st.title("ASK ME ANYTHING")
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest',temperature=0.6)
main_placeholder = st.empty()

uploaded_file  = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file :
    with st.spinner("Data Loading..."):
        try:
            loader = PyPDFLoader("transformer.pdf")
            data = loader.load()
            main_placeholder.text("Data Loaded Successfully...✅✅✅")
        except Exception as e:
            main_placeholder.text(f"Error loading data: {e}")

    with st.spinner("Splitting Text..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n","\n"," "],
                chunk_size=1000,
                chunk_overlap=200
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
        except Exception as e:
            main_placeholder.text(f"Error splitting text: {e}")

    with st.spinner("Building Embedding Vector..."):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Built...✅✅✅")
            time.sleep(2)
        except Exception as e:
            main_placeholder.text(f"Error building embedding vector: {e}")

    query = main_placeholder.text_input("Question: ")
    if query:
        chain = load_qa_chain(llm=model,chain_type="stuff")
        result = chain.run(input_documents=docs,question=query)
        st.write(result)