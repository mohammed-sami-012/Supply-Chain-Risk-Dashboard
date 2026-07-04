import asyncio
from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

loader = Docx2txtLoader("Research_Paper.docx")
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    question: str
    route: str
    answer: str


def router_node(state: AgentState) -> AgentState:
    prompt = f"""Classify into one word: "document" or "data".
"document": research findings, methodology, model performance, general concepts.
"data": specific Order ID or live counts.

Question: {state['question']}
Answer with just one word."""
    result = llm.invoke(prompt).content.strip().lower()
    state["route"] = "data" if "data" in result else "document"
    return state


def rag_node(state: AgentState) -> AgentState:
    relevant = vector_store.similarity_search(state["question"], k=3)
    context = "\n\n".join(d.page_content for d in relevant)
    prompt = f"Answer using only this context. If not found, say you don't know.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
    state["answer"] = llm.invoke(prompt).content
    return state


async def build_data_node():
    # This launches mcp_server.py as a subprocess and connects to it
    client = MultiServerMCPClient({
        "supply_chain": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()          # discovers get_order_risk & get_high_risk_count
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    async def data_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state["question"])
        if response.tool_calls:
            call = response.tool_calls[0]
            tool = tool_map[call["name"]]
            result = await tool.ainvoke(call["args"])

            if isinstance(result, list):
                # MCP returns a list of content blocks like {'type': 'text', 'text': '...'}
                state["answer"] = "\n".join(
                    item.get("text", str(item)) for item in result if isinstance(item, dict)
                )
            else:
                state["answer"] = str(result)
        else:
            state["answer"] = response.content
        return state

    return data_node


async def main():
    data_node = await build_data_node()

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("data", data_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda s: s["route"], {"document": "rag", "data": "data"})
    graph.add_edge("rag", END)
    graph.add_edge("data", END)
    app = graph.compile()

    print("MCP-powered Supply Chain Assistant (type 'exit' to quit)")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        result = await app.ainvoke({"question": q, "route": "", "answer": ""})
        print("Bot:", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())