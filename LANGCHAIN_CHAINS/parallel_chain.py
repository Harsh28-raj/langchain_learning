from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel 
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Generate short summary of interesting facts about {input} and return in JSON format",
    input_variables=["input"]
)

prompt2 = PromptTemplate(
    template="Generate 5 interesting questions related to {input} and return in JSON format",
    input_variables=["input"]
)

prompt3 = PromptTemplate(
    template="""
Merge these notes into one summary:

Notes 1:
{notes1}

Notes 2:
{notes2}

Return final summary.
""",
    input_variables=["notes1","notes2"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes1': prompt1 | model1 | parser,
    'notes2': prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"input": "cricket"})
print(result)

chain.get_graph().print_ascii()