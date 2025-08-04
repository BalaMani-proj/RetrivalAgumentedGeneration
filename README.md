# AI Chatbot

This project is an AI-powered chatbot application built with Gradio, HuggingFace Transformers, and LangChain. It leverages Retrieval-Augmented Generation (RAG) to answer user questions using a local knowledge base stored in a Chroma vector database.

## Features
- **Gradio UI**: Simple and interactive web interface for chatting with the AI.
- **Retrieval-Augmented Generation (RAG)**: Combines semantic search with generative AI to provide contextually relevant answers.
- **Local Vector Database**: Uses Chroma for fast document retrieval.
- **HuggingFace Transformers**: Utilizes state-of-the-art language models for response generation.
- **Streaming Responses**: Answers are streamed progressively for a better user experience.

## Directory Structure
```
├── src/                  # Source code (main: mygradio.py)
├── chroma_db/            # Chroma vector database files
├── tessdata/             # Tesseract data (if OCR is used)
├── doc/                  # Documentation and licenses
├── ...                   # Other binaries, DLLs, and support files
```

## Setup & Installation
1. **Clone the repository**
   ```powershell
   git clone <your-repo-url>
   cd insenv
   ```
2. **Create and activate a Python virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Run tests**
   ```powershell
   python -m unittest discover tests
   ```
   - Required packages include: `gradio`, `torch`, `transformers`, `langchain_chroma`, `langchain_huggingface`, `sentence-transformers`, etc.

4. **Download or prepare your knowledge base**
   - Place your documents in the appropriate directory and ensure they are indexed in Chroma.

## Usage
Run the Gradio chatbot UI:
```powershell
python src\mygradio.py
```
- Open the provided local URL in your browser to interact with the chatbot.

## Docker Deployment
To build and run the chatbot in Docker:
```powershell
docker build -t chatbot .
docker run -p 7860:7860 chatbot
```
The app will be available at `http://localhost:7860`.

## How It Works
- User submits a question via the Gradio interface.
- The app retrieves relevant documents from the Chroma vector store using semantic search.
- The retrieved context is combined with the user's question and passed to a HuggingFace language model.
- The model generates a response, which is streamed back to the user.

## Customization
- **Model Selection**: Change the model name in `mygradio.py` to use different HuggingFace models.
- **Knowledge Base**: Add or update documents in the Chroma database to improve answer quality.

## Industry Applications
This chatbot is designed to be adaptable for use across multiple industries, including:

- **Customer Support**: Automate responses to common queries, provide instant help, and reduce support workload.
- **Healthcare**: Answer patient FAQs, assist with appointment scheduling, and provide information from medical knowledge bases.
- **Education**: Serve as a virtual tutor, answer student questions, and deliver personalized learning content.
- **Finance**: Respond to client inquiries, explain financial products, and assist with onboarding processes.
- **Legal**: Retrieve relevant case law, answer legal questions, and support document review.
- **Human Resources**: Help employees with HR policies, benefits, and onboarding information.
- **IT & Tech**: Provide troubleshooting steps, technical documentation, and developer support.

The modular architecture allows you to customize the knowledge base and language model for your specific domain, making it suitable for a wide range of business and research applications.

## License
See `doc/LICENSE` for license details.

## Acknowledgements
- [Gradio](https://gradio.app/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [LangChain](https://langchain.com/)
- [Chroma](https://www.trychroma.com/)
