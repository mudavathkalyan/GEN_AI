from fastapi import FastAPI
from langchain.prompt import ChatPromptTemplete
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os
from langchain_community.llms import Ollama


from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()
#