import gradio as gr
import torch
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
import time
import json  # Import json to handle JSON formatting

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer_qa = AutoTokenizer.from_pretrained(model_name)
model_qa = AutoModelForCausalLM.from_pretrained(model_name)

def generate_with_rag(query):
   
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return json.dumps({"response": "No relevant documents found."})
    
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Construct the prompt based on user input
    prompt = (
        f"Context: {context}\n"
        "Answer:"
    )
    
    inputs = tokenizer_qa(prompt, return_tensors="pt")
    
    output_tokens = model_qa.generate(
        **inputs, 
        max_new_tokens=500,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    
    answer = tokenizer_qa.decode(output_tokens[0], skip_special_tokens=True)
    
    # Stream output progressively
    for i in range(1, len(answer) + 1):
        yield gr.update(value=answer[:i])
        time.sleep(0.05)  # Faster response streaming

    # Return the answer in JSON format
    response = {
        "query": query,        
        "generated_answer": answer
    }
    
    return json.dumps(response, indent=4)  # Format the response as JSON

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            gr.HTML("<h1 style='text-align: center;'>Bala's Chatbot</h1>")
            
            query_input = gr.Textbox(label="Ask a question:", placeholder="Enter your question here", lines=1)
            #prompt_input = gr.Textbox(label="Enter prompt", placeholder="Enter a custom prompt here", lines=1)
            output_text = gr.Textbox(label="Response", interactive=False, lines=15)  # Adjusted lines for JSON response
            
            query_input.submit(generate_with_rag, inputs=[query_input], outputs=output_text)

# Launch Gradio UI
demo.launch()
