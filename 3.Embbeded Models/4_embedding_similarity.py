from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents=[
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Madrid is the capital of Spain",
    "Rome is the capital of Italy"
]

query = "What is the capital of France?"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

print("Similarities:", similarities)

# get most similar document
index = np.argmax(similarities)
print("Most similar document:", documents[index])
index,score=sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[0]