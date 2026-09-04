from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

os.environ["GEMMA_APIKEY"] = os.getenv("GEMMA_APIKEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(title="Langchain Demo with GEMINI")

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", api_key=os.getenv("GEMMA_APIKEY"),
    generation_config={
        "thinking_budget": 0  
    })

prompt = ChatPromptTemplate.from_template("What is {topic}?")

add_routes(
    app,
    prompt | llm,
    path="/chat"
)

add_routes(
    app,llm,
    path="/any"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost",port=8000)