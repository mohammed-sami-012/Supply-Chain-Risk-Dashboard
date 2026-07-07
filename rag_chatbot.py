from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Step A: Load the document ---
loader = Docx2txtLoader("Research_Paper.docx")
documents = loader.load()

# --- Step B: Split it into small chunks ---
# Why: embedding a whole 5-page document as one block loses precision.
# Smaller chunks let us retrieve just the relevant paragraph, not the whole paper.
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# --- Step C: Embed each chunk and store in a searchable vector index ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# --- Step D: Set up the LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

print("RAG Assistant — ask about the Research Paper (type 'exit' to quit)")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # --- Step E: Retrieve the most relevant chunks for this question ---
    relevant_chunks = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    # --- Step F: Ask the LLM to answer using ONLY the retrieved context ---
    prompt = f"""Answer the question using only the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}
"""
    response = llm.invoke(prompt)
    print("Bot:", response.content)