import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

load_dotenv()

# 1. Manual Model & Tokenizer loading (Zyada stable hai)
model_id = "google/flan-t5-base"

@st.cache_resource # Taki model baar baar load na ho
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # torch_dtype=torch.float32 use kar rahe hain meta-tensor error se bachne ke liye
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    pipe = pipeline(
        "text2text-generation", # Agar ye error de, toh "text-generation" kar dena
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256,
        device=-1 # -1 matlab CPU, agar GPU hai toh 0 karein
    )
    return HuggingFacePipeline(pipeline=pipe)

try:
    llm = load_model()
except Exception as e:
    st.error(f"Model load karne mein error: {e}")
    st.stop()

st.header("Research Tool")

# Input fields
paper_input = st.selectbox("Select Research Paper", ["Attention Is All You Need", "BERT", "GPT-3"])
style_input = st.selectbox("Style", ["Beginner-Friendly", "Technical"])
length_input = st.selectbox("Length", ["Short", "Long"])

template_text = """Summarize this paper: {paper_input}
Style: {style_input}
Length: {length_input}
Summary:"""

template = PromptTemplate(input_variables=["paper_input", "style_input", "length_input"], template=template_text)

if st.button("Summarize"):
    with st.spinner("Processing..."):
        chain = template | llm
        result = chain.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })
        st.subheader("Result:")
        st.write(result)