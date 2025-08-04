import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set the correct database path (ensure it's not inside venv)
db_path = os.path.join(os.getcwd(), "chroma_db")  # Ensures the path is relative to the project

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma with the correct path
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)

print(f"Chroma DB is being stored at: {db_path}")