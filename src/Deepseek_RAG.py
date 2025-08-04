import torch
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer




# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma with the embedding function
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Initialize the retriever
retriever = vectorstore.as_retriever(search_type="mmr", k=1)

# Check number of documents in Chroma DB
num_docs = vectorstore._collection.count()
print(f"Number of documents in Chroma DB: {num_docs}")

retrieved_sample = vectorstore._collection.peek(limit=1)
print(f"Sample document from Chroma DB: {retrieved_sample}")

# Use DeepSeek-Coder-1.3B-Instruct
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load tokenizer
tokenizer_qa = AutoTokenizer.from_pretrained(model_name)

# Load model without quantization (for CPU)
model_qa = AutoModelForCausalLM.from_pretrained(model_name)

def generate_with_rag(query):
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)  # ✅ FIXED

    if not retrieved_docs:
        return "No relevant documents found."

    # Combine the retrieved document content
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Format input for DeepSeek
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer_qa(prompt, return_tensors="pt")

    # Step 3: Generate response
    output_tokens = model_qa.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=0.9)
    answer = tokenizer_qa.decode(output_tokens[0], skip_special_tokens=True)

    return answer

# Example usage
query = "What is the company name in the professional summary?"
response = generate_with_rag(query)
print(response)

#query_2 = "What is the fruit name?"
#response_2 = generate_with_rag(query_2)
#print(response_2)
