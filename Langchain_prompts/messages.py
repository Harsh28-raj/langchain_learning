import os
from dotenv import load_dotenv
# OpenAI ki jagah Groq ka import
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# Key fetch karo (Jo tune .env mein rakhi hai)
api_key = os.getenv("GROQ_API_KEY")

# Model initialize (Groq use kar rahe hain kyunki OpenAI paid hai)
model = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

messages = [
    SystemMessage(content="You are a helpful assistant that provides information about the capital cities of countries."),
    HumanMessage(content="Tell me about LangChain?"),
]

# Ab invoke makkhan chalega
result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

# Final Output print
print("--- Chat History ---")
for msg in messages:
    print(f"{type(msg).__name__}: {msg.content}")