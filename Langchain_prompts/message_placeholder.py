from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant that provides information about {domain}.'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', 'Can you explain {topic} to me?'),])

chat_history = []
#load chat history

with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())
    

print(chat_history)    

# create prompt
chat_template.invoke({"domain": "AI", "topic": "machine learning", "chat_history": chat_history})



