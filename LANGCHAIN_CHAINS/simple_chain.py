from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {input} and return in JSON format",
    input_variables=["input"]
)

model = ChatGroq(
    model="llama3-8b-8192"   # groq supported model
)

parser = JsonOutputParser()

chain = prompt | model | parser

result = chain.invoke({"input": "cricket"})

print(result)

chain.get_graph().print_ascii()