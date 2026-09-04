from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GEMMA_APIKEY"] = os.getenv("GEMMA_APIKEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an Assistant"),
        ("user","Question: {question}")
    ]
)

st.title("Langchain Demo with GEMMA")
input_text = st.text_input("Search")


llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash",api_key=os.getenv("GEMMA_APIKEY"))
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))