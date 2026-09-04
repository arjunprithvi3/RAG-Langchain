import requests
import streamlit as st

def get_response(input_text):

    response = requests.post("http://localhost:8000/chat/invoke", json={"input": {"topic": input_text}})
    return response.json()['content'][0]['text']


st.title("Langchain Demo with GEMINI")
input_text = st.text_input("Search")

if input_text:
    st.write(get_response(input_text))


def get_any_response(input):

    response = requests.post("http://localhost:8000/any/invoke", json={"input": input})
    return response.json()['output']

st.title("Ask Anything with GEMINI")
input = st.text_input("Ask Anything")

if input:
    st.write(get_any_response(input))