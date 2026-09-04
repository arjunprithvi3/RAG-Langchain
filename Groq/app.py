import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

load_dotenv()

groq_api = os.environ['GROQ_API']

if "messages" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader("https://reference.langchain.com/python/langchain")
    st.session_state.docs =  st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    st.session_state.final_doc =  st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings)

st.title("Chat Groq Demo")
llm = ChatGroq(groq_api_key=groq_api,model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
   "Use the following context to answer the question.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retrival_chain = create_retrieval_chain(retriever,document_chain)

query = st.text_input("input your propt here")

if query:
    start = time.process_time()
    response = retrival_chain.invoke({"context": "retrieved docs go here","question": query})
    print("Response time:",time.process_time() - start)
    st.write(response['answer'])

