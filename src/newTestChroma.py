# Load existing ChromaDB
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings and vector store correctly
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Get number of stored documents
collection_count = vectorstore._collection.count()
print(f"Total documents stored in ChromaDB: {collection_count}")