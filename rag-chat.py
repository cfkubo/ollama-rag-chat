from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import gradio as gr
from ollama import Client

class TimeoutClient(Client):
    """
    A simple wrapper for the Ollama client to set a timeout.
    This helps prevent the application from hanging on long requests.
    """
    def __init__(self, host="http://127.0.0.1:11434", timeout=600):
        super().__init__(host=host)
        self._client.timeout = timeout

# Configure Ollama LLM
# You can change the model name to any model you have pulled in Ollama.
# Example: "llama2:latest", "mistral:latest", "gemma:latest"
ollama_llm = Ollama(model="llama2:latest", client=TimeoutClient())

# Configure Embedding Model
# This model is used to create numerical representations (embeddings) of your documents.
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Configure LlamaIndex settings
# This sets the global LLM and embedding model for LlamaIndex to use.
Settings.llm = ollama_llm
Settings.embed_model = embed_model

def answer(message, history):
    """
    Handles user messages, either performing RAG on uploaded files or
    generating a response directly from the LLM.
    """
    if "files" in message and message["files"]:
        # If files are uploaded, use them for RAG
        files = [file for file in message["files"]]
        documents = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return str(query_engine.query(message["text"]))
    else:
        # If no files, just use the LLM to complete the prompt
        return Settings.llm.complete(message["text"]).text

# Create the Gradio chat interface
demo = gr.ChatInterface(
    answer,
    type="messages",
    title="Ollama RAG Chatbot",
    description="Upload documents to chat with them, or chat without files.",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".md", ".docx", ".html"]),
    multimodal=True,
)

# Launch the Gradio application
if __name__ == "__main__":
    demo.launch()