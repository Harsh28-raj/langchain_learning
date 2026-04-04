from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typer import prompt

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant that provides information about {domain}.'),
    ('human', 'Can you explain {topic} to me?')
])

prompt = chat_template.invoke({"domain": "AI", "topic": "machine learning"})

print(prompt)