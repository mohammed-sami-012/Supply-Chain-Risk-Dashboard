import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

# --- Delay Risk Overview ---

st.set_page_config(page_title="Delay Risk Dashboard", layout="wide")


# @st.cache_data avoids reloading the CSV on every user interaction — good practice for perf
@st.cache_data
def load_data():
    return pd.read_csv('all_order_risk_results.csv')


df = load_data()

# --- Sidebar: User Capabilities (filters) ---
st.sidebar.header("Filters")

shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    options=df['Shipping Mode'].unique().tolist(),
    default=df['Shipping Mode'].unique().tolist()
)

regions = st.sidebar.multiselect(
    "Order Region",
    options=df['Order Region'].unique().tolist(),
    default=df['Order Region'].unique().tolist()
)

segments = st.sidebar.multiselect(
    "Customer Segment",
    options=df['Customer Segment'].unique().tolist(),
    default=df['Customer Segment'].unique().tolist()
)

risk_threshold = st.sidebar.slider(
    "Minimum Late Delivery Probability",
    min_value=0.0, max_value=1.0, value=0.0, step=0.01
)

df = df[
    (df['Shipping Mode'].isin(shipping_modes)) &
    (df['Order Region'].isin(regions)) &
    (df['Customer Segment'].isin(segments)) &
    (df['Late Delivery Probability'] >= risk_threshold)
]

if df.empty:
    st.warning("No orders match the current filter selection.")
    st.stop()

st.title("Late Delivery Risk Dashboard")


# ==========================================================
# --- AI Assistant: LangChain + RAG + LangGraph + MCP ---
# ==========================================================

class AgentState(TypedDict):
    question: str
    route: str
    answer: str


@st.cache_resource
def build_agent():
    loader = Docx2txtLoader("Research_Paper.docx")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

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
        prompt = (f"Answer using only this context. If not found, say you don't know.\n\n"
                  f"Context:\n{context}\n\nQuestion: {state['question']}")
        state["answer"] = llm.invoke(prompt).content
        return state

    import sys

    async def setup():
        client = MultiServerMCPClient({
            "supply_chain": {"command": sys.executable, "args": ["mcp_server.py"], "transport": "stdio"}
        })
        tools = await client.get_tools()
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        async def data_node(state: AgentState) -> AgentState:
            response = llm_with_tools.invoke(state["question"])
            if response.tool_calls:
                call = response.tool_calls[0]
                result = await tool_map[call["name"]].ainvoke(call["args"])
                if isinstance(result, list):
                    state["answer"] = "\n".join(
                        item.get("text", str(item)) for item in result if isinstance(item, dict)
                    )
                else:
                    state["answer"] = str(result)
            else:
                state["answer"] = response.content
            return state

        graph = StateGraph(AgentState)
        graph.add_node("router", router_node)
        graph.add_node("rag", rag_node)
        graph.add_node("data", data_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges("router", lambda s: s["route"], {"document": "rag", "data": "data"})
        graph.add_edge("rag", END)
        graph.add_edge("data", END)
        return graph.compile()

    return asyncio.run(setup())


agent_app = build_agent()

st.header("Ask the Assistant")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask about the report or a specific order...")

if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(agent_app.ainvoke({"question": user_question, "route": "", "answer": ""}))
            st.write(result["answer"])

    st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"]})


# ==========================================================
# --- Delay Risk Overview ---
# ==========================================================

st.header("Delay Risk Overview")

# --- Overall risk distribution ---
st.subheader("Overall Risk Distribution")
risk_counts = df['Risk Category'].value_counts()
st.bar_chart(risk_counts)

# --- High-risk order count ---
high_risk_count = (df['Risk Category'] == 'High Risk').sum()
total_orders = len(df)

col1, col2 = st.columns(2)
col1.metric("High-Risk Orders", high_risk_count)
col2.metric("% of Total Orders", f"{high_risk_count / total_orders:.1%}")

# ==========================================================
# --- Order-Level Risk Prediction ---
# ==========================================================

st.header("Order-Level Risk Prediction")

order_id = st.number_input(
    "Enter Order ID to inspect",
    min_value=int(df['Order ID'].min()),
    max_value=int(df['Order ID'].max()),
    step=1
)

if st.button("Get Risk Details"):
    order_row = df[df['Order ID'] == order_id]

    if order_row.empty:
        st.warning("No order found with that ID.")
    else:
        order_row = order_row.iloc[0]
        prob = order_row['Late Delivery Probability']
        category = order_row['Risk Category']
        drivers = order_row['Key Risk Drivers']

        # --- Individual order risk score ---
        col1, col2 = st.columns(2)
        col1.metric("Late Delivery Probability", f"{prob:.1%}")
        col2.write(f"**Risk Category**")

        if category == 'Low Risk':
            col2.success(category)
        elif category == 'Medium Risk':
            col2.warning(category)
        else:
            col2.error(category)

        st.progress(prob)

        # --- Key contributing factors ---
        st.subheader("Key Contributing Factors")
        for factor in drivers.split(', '):
            st.write(f"- {factor}")

        with st.expander("Full Order Details"):
            st.write(order_row[['Order Region', 'Customer Segment', 'Product Name', 'Shipping Mode']])

# One usability addition
st.caption("Example High Risk Order IDs to try:")
st.write(df[df['Risk Category'] == 'High Risk']['Order ID'].head(10).tolist())

# ==========================================================
# --- Region & Mode Risk Analysis ---
# ==========================================================

st.header("Region & Mode Risk Analysis")

st.subheader("Risk Heatmap by Region")

region_risk = pd.crosstab(df['Order Region'], df['Risk Category'], normalize='index') * 100
region_risk = region_risk.reindex(columns=['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0)

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(region_risk, annot=True, fmt='.1f', cmap='Reds', ax=ax)
ax.set_xlabel("Risk Category")
ax.set_ylabel("Order Region")
st.pyplot(fig)

# Shipping Mode Risk Comparison
st.subheader("Shipping Mode Risk Comparison")

mode_risk = pd.crosstab(df['Shipping Mode'], df['Risk Category'], normalize='index') * 100
mode_risk = mode_risk.reindex(columns=['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0)

st.bar_chart(mode_risk)

worst_region = region_risk['High Risk'].idxmax()
worst_mode = mode_risk['High Risk'].idxmax()

st.caption(f"Highest High-Risk rate by region: **{worst_region}** ({region_risk['High Risk'].max():.1f}%)")
st.caption(f"Highest High-Risk rate by shipping mode: **{worst_mode}** ({mode_risk['High Risk'].max():.1f}%)")

# ==========================================================
# --- Operations Action Panel ---
# ==========================================================

st.header("Operations Action Panel")

st.subheader("Orders Requiring Immediate Attention")

col1, col2 = st.columns(2)

with col1:
    selected_categories = st.multiselect(
        "Risk Category",
        options=['Low Risk', 'Medium Risk', 'High Risk'],
        default=['High Risk']
    )

with col2:
    min_prob = st.slider(
        "Minimum Late Delivery Probability",
        min_value=0.0, max_value=1.0, value=0.66, step=0.01
    )

action_queue = df[
    (df['Risk Category'].isin(selected_categories)) &
    (df['Late Delivery Probability'] >= min_prob)
]

st.write(f"**{len(action_queue)} orders** currently meet this criteria")

# Risk-Based Prioritization
st.subheader("Risk-Based Prioritization")

action_queue_sorted = action_queue.sort_values('Late Delivery Probability', ascending=False)

st.dataframe(
    action_queue_sorted[['Order ID', 'Late Delivery Probability', 'Risk Category',
                          'Key Risk Drivers', 'Order Region', 'Shipping Mode', 'Product Name']],
    use_container_width=True
)

csv_data = action_queue_sorted.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Action Queue as CSV",
    data=csv_data,
    file_name="operations_action_queue.csv",
    mime="text/csv"
)