from dotenv import load_dotenv
import os
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Setup: same RAG pieces as Stage 2 ---
loader = Docx2txtLoader("Research_Paper.docx")
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# --- Setup: the order data ---
orders_df = pd.read_csv("all_order_risk_results.csv")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))


# --- Step A: Define the shared state ---
class AgentState(TypedDict):
    question: str
    route: str
    answer: str


# --- Step B: The router node — decides which path to take ---
def router_node(state: AgentState) -> AgentState:
    prompt = f"""Classify this question into exactly one word: "document" or "data".
- "document": questions about research findings, methodology, model performance, general concepts.
- "data": questions about a specific Order ID, or live counts (e.g. how many high risk orders exist).

Question: {state['question']}
Answer with just one word: document or data."""
    result = llm.invoke(prompt).content.strip().lower()
    state["route"] = "data" if "data" in result else "document"
    return state


# --- Step C: The RAG node (Stage 2's logic) ---
def rag_node(state: AgentState) -> AgentState:
    relevant_chunks = vector_store.similarity_search(state["question"], k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"""Answer using only this context. If not found, say you don't know.

Context:
{context}

Question: {state['question']}"""
    state["answer"] = llm.invoke(prompt).content
    return state


# --- Step D: The data node — looks up live order info ---
def data_node(state: AgentState) -> AgentState:
    question = state["question"]
    digits = "".join(filter(str.isdigit, question))

    if digits and int(digits) in orders_df["Order ID"].values:
        order = orders_df[orders_df["Order ID"] == int(digits)].iloc[0]
        state["answer"] = (
            f"Order {digits}: {order['Late Delivery Probability']:.1%} late delivery probability, "
            f"Risk Category: {order['Risk Category']}, Key Drivers: {order['Key Risk Drivers']}"
        )
    else:
        high_risk_count = (orders_df["Risk Category"] == "High Risk").sum()
        state["answer"] = f"There are currently {high_risk_count} orders flagged High Risk in the dataset."
    return state


# --- Step E: Build the graph ---
graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("data", data_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda state: state["route"], {"document": "rag", "data": "data"})
graph.add_edge("rag", END)
graph.add_edge("data", END)

app = graph.compile()

# --- Step F: Chat loop ---
print("Agentic Supply Chain Assistant (type 'exit' to quit)")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    result = app.invoke({"question": q, "route": "", "answer": ""})
    print("Bot:", result["answer"])