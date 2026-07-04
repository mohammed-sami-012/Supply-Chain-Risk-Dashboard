from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

# This list is our "memory" — every message, from both sides, gets added here
chat_history = [
    SystemMessage(content="You are a helpful assistant that answers questions about supply chain "
                           "operations and late delivery risk. Be concise and clear.")
]

print("Supply Chain Risk Assistant (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))

    print("Bot:", response.content)