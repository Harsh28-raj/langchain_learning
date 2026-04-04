import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# 1. Load Environment Variables
load_dotenv()

# Key check
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY nahi mili! .env file check karein.")
    exit()

# 2. Setup Groq Model
try:
    model = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
    print("✅ Groq (Llama-3) is ready to roll!")
except Exception as e:
    print(f"❌ Setup Error: {e}")
    exit()

chat_history = [
    SystemMessage(content='You are a helpful assistant that provides information')
]

# 3. Chat Loop
while True:
    try:
        user_input = input("\nYou: ")
        chat_history.append(HumanMessage(content=user_input))
        if user_input.lower() in ["exit", "quit"]:
            print("Bbye!")
            break
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))    
        if not user_input.strip():
            continue

        # History mein user ka input dalo
        chat_history.append(user_input)

        # Sirf ek baar invoke karo poori history ke saath
        response = model.invoke(chat_history)
        
        # AI ka sirf text (content) history mein dalo
        chat_history.append(response.content)
        
        print(f"\nAI: {response.content}")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        
print("\n--- Full Chat History ---")
print(chat_history)