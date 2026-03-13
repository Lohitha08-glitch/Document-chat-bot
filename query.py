from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k":3})

# Load lightweight local LLM
llm = OllamaLLM(model="tinyllama")

# Ask user question
query = input("Ask a question: ")

# Retrieve relevant document chunks
docs = retriever.invoke(query)

# Combine retrieved text into context
context = "\n\n".join([doc.page_content for doc in docs])

# Create prompt
prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

# Generate answer
response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)