from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from src.chroma_utils import vectorstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Safe defaults for env-configurable values
DEFAULT_RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

output_parser = StrOutputParser()

# Setting up prompts
contextualize_q_system_prompt = os.getenv("CONTEXTUALIZE_Q_PROMPT", 
    "Given a chat history and the latest user question " 
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question in detail."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="gemini-2.0-flash"):
    # Use the latest model names supported by Google Generative AI
    allowed_models = {
        "gemini-2.0-flash",
        "gemini-2.0-pro"
    }

    model_to_use = model if model in allowed_models else "gemini-2.0-flash"

    llm = ChatGoogleGenerativeAI(
        model=model_to_use,
        google_api_key=GOOGLE_API_KEY,
        temperature=DEFAULT_TEMPERATURE,
         transport="rest",  
        api_version="v1"
    )

    # Build retriever at call-time
    retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVER_K})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
