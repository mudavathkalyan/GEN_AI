from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

## Streamlit UI
st.title('Langchain With tinyllama ')
input_text = st.text_input("Ask anything you wish for ...")

## Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

## Load Local Ollama Model (No API Key Needed)
llm = Ollama(model="tinyllama")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

## Generate Response
if input_text:
    st.write(chain.invoke({"question": input_text}))
