from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Step 1: Load the PDF
pdf_path = r"C:\projects\PC-AppGuide.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Step 3: Define Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use the model for embeddings

# Step 4: Add texts and embeddings to Chroma DB
vectorstore = Chroma.from_texts(
    [doc.page_content for doc in texts],  # Extract page content
    embeddings,  # Pass the embeddings object directly
    persist_directory="./chroma_db"  # Chroma will automatically handle persistence
)

# No need for manual persist() anymore
print("PDF loaded and stored in ChromaDB successfully!")