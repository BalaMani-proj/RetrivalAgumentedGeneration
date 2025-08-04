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
retriever = vectorstore.as_retriever(search_type="mmr", k=3)  # Increased k to 3

# Check number of documents in Chroma DB
num_docs = vectorstore._collection.count()
print(f"Number of documents in Chroma DB: {num_docs}")

if num_docs == 0:
    print("⚠️ No documents found in ChromaDB!")

retrieved_sample = vectorstore._collection.peek(limit=5)
print(f"Sample document from Chroma DB: {retrieved_sample}")

# Use DeepSeek-Coder-1.3B-Instruct
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load tokenizer
tokenizer_qa = AutoTokenizer.from_pretrained(model_name)

# Load model without quantization (for CPU)
model_qa = AutoModelForCausalLM.from_pretrained(model_name)

def generate_with_rag(query):
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # If no relevant documents found
    if not retrieved_docs:
        return "No relevant documents found."

    # Debugging: Print retrieved documents for clarity
    print("\n=== Retrieved Documents ===")
    for idx, doc in enumerate(retrieved_docs):
        print(f"Doc {idx + 1}: {doc.page_content}")

    # Combine the retrieved document content
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Format input for DeepSeek
    prompt = (
        "You are an AI assistant tasked with extracting key information from documents.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # Debugging: Print the generated prompt
    print("\n=== Generated Prompt Sent to Model ===")
    print(prompt)

    # Tokenize and generate response
    inputs = tokenizer_qa(prompt, return_tensors="pt")
    output_tokens = model_qa.generate(
        **inputs, 
        max_new_tokens=50,  # Limit length
        do_sample=True,    # Deterministic output
        temperature=0.6,    # Reduce randomness
        top_p=0.9           # Limit randomness
    )

    answer = tokenizer_qa.decode(output_tokens[0], skip_special_tokens=True)

    return answer

# Example usage
#query = "What is the company name in the professional summary?"
#response = generate_with_rag(query)
#print(response)

# You can test more queries as needed
query_2 = "What is your area of expertise?"
response_2 = generate_with_rag(query_2)
print(response_2)
