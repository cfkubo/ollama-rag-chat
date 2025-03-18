from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import gradio as gr
from ollama import Client
import logging
import os
import requests
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeoutClient(Client):
    def __init__(self, host="http://ollama-rag-chat-ollama-1:11434", timeout=600):
        super().__init__(host=host)
        self._client.timeout = timeout

def get_running_ollama_model():
    try:
        logging.info("Attempting to retrieve running Ollama model from Ollama API...")
        response = requests.get("http://ollama-rag-chat-ollama-1:11434/api/tags")
        response.raise_for_status()
        data = response.json()
        if "models" in data and data["models"]:
            model_names = [model["name"] for model in data["models"]]
            if model_names:
                model_name = model_names[0]
                logging.info(f"Detected running Ollama model: {model_name}")
                return model_name
        logging.warning("No Ollama models found via API.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting running Ollama model from Ollama API: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

# Authenticate with Hugging Face using environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    logging.info("Attempting to log in to Hugging Face using environment variable...")
    try:
        login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)
        logging.info("Successfully logged in to Hugging Face.")
    except Exception as e:
        logging.error(f"Error logging in to Hugging Face: {e}")
else:
    logging.warning("HUGGINGFACE_TOKEN environment variable not set. Public models will be used.")

# Get the running Ollama model
running_model = get_running_ollama_model()

# Configure Ollama LLM
if running_model:
    ollama_model = running_model
    logging.info(f"Using running Ollama model: {ollama_model}")
else:
    ollama_model = "llama2:latest"
    logging.warning(f"No running Ollama model found. Using default: {ollama_model}")

ollama_llm = Ollama(model=ollama_model, client=TimeoutClient())

# Configure Embedding Model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Configure LlamaIndex Settings
Settings.llm = ollama_llm
Settings.embed_model = embed_model

# def answer(message, history):
#     files_to_process = []

#     # Handle newly uploaded files
#     if "files" in message and message["files"]:
#         for file in message["files"]:
#             files_to_process.append(file.name)  # Get the path of the temporary file

#     # Handle files from history
#     for msg in history:
#         if msg['role'] == "user":
#             if isinstance(msg['content'], tuple):
#                 files_to_process.append(msg['content'][0])  # Get the path from history
#             elif isinstance(msg['content'], list):
#                 for item in msg['content']:
#                     if isinstance(item, tuple):
#                         files_to_process.append(item[0])

#     if files_to_process:
#         accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
#         filtered_files = [f for f in files_to_process if os.path.splitext(f)[1].lower() in accepted_files]

#         if filtered_files:
#             documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
#             index = VectorStoreIndex.from_documents(documents)
#             query_engine = index.as_query_engine()
#             return str(query_engine.query(message["text"]))
#         else:
#             return "No accepted file types uploaded."
#     else:
#         return Settings.llm.complete(message["text"]).text

def answer(message, history):
    files_to_process = []

    # Handle newly uploaded file (only the first one if multiple are sent)
    if "files" in message and message["files"]:
        if isinstance(message["files"][0], dict) and "file" in message["files"][0]:
            files_to_process.append(message["files"][0]["file"]["path"])
        elif hasattr(message["files"][0], "name"):
            files_to_process.append(message["files"][0].name)

    # Handle file from history (only the last one if present)
    for msg in reversed(history):  # Check history in reverse order to get the latest
        if msg['role'] == "user":
            if isinstance(msg['content'], tuple):
                files_to_process.append(msg['content'][0])
                break  # Only consider the latest uploaded file from history
            elif isinstance(msg['content'], list):
                for item in reversed(msg['content']):
                    if isinstance(item, tuple):
                        files_to_process.append(item[0])
                        break
                if files_to_process:
                    break
            elif isinstance(msg['content'], dict) and "file" in msg['content']:
                files_to_process.append(msg['content']['file']['path'])
                break
            elif isinstance(msg['content'], list):
                for item in reversed(msg['content']):
                    if isinstance(item, dict) and "file" in item:
                        files_to_process.append(item['file']['path'])
                        break
                if files_to_process:
                    break

    if files_to_process:
        accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
        filtered_files = [f for f in files_to_process if os.path.splitext(f)[1].lower() in accepted_files]

        if filtered_files:
            documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            return str(query_engine.query(message["text"]))
        else:
            return "No accepted file types uploaded."
    else:
        return Settings.llm.complete(message["text"]).text
    
# demo = gr.ChatInterface(
#     answer,
#     type="messages",
#     title="Ollama Llama Index RAG Chatbot",
#     description="Upload any text,pdf,json,html,docx,csv files and ask questions about them, or chat without files.",
#     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
#     multimodal=True,
# )

demo = gr.ChatInterface(
    answer,
    type="messages",
    title="Ollama Llama Index RAG Chatbot",
    description="Upload one text, pdf, json, csv, markdown, html, or docx file and ask questions about it, or chat without files.",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
    multimodal=True,
)


demo.launch(server_name="0.0.0.0", server_port=7860)


# ````
# Traceback (most recent call last):
#   File "/usr/local/lib/python3.9/site-packages/gradio/queueing.py", line 536, in process_events
#     response = await route_utils.call_process_api(
#   File "/usr/local/lib/python3.9/site-packages/gradio/route_utils.py", line 322, in call_process_api
#     output = await app.get_blocks().process_api(
#   File "/usr/local/lib/python3.9/site-packages/gradio/blocks.py", line 1935, in process_api
#     result = await self.call_function(
#   File "/usr/local/lib/python3.9/site-packages/gradio/blocks.py", line 1518, in call_function
#     prediction = await fn(*processed_input)
#   File "/usr/local/lib/python3.9/site-packages/gradio/utils.py", line 793, in async_wrapper
#     response = await f(*args, **kwargs)
#   File "/usr/local/lib/python3.9/site-packages/gradio/chat_interface.py", line 623, in _submit_fn
#     response = await anyio.to_thread.run_sync(
#   File "/usr/local/lib/python3.9/site-packages/anyio/to_thread.py", line 56, in run_sync
#     return await get_async_backend().run_sync_in_worker_thread(
#   File "/usr/local/lib/python3.9/site-packages/anyio/_backends/_asyncio.py", line 2470, in run_sync_in_worker_thread
#     return await future
#   File "/usr/local/lib/python3.9/site-packages/anyio/_backends/_asyncio.py", line 967, in run
#     result = context.run(func, *args)
#   File "/app/rag-ollama.py", line 82, in answer
#     files_to_process.append(file.name)  # Get the path of the temporary file
# AttributeError: 'dict' object has no attribute 'name'
# ```


# # def answer(message, history):
# #     files = []
# #     if "files" in message and message["files"]:  # Check if files are uploaded
# #         for msg in history:
# #             if msg['role'] == "user" and isinstance(msg['content'], tuple):
# #                 files.append(msg['content'][0])
# #         for file in message["files"]:
# #             files.append(file)

# #         documents = SimpleDirectoryReader(input_files=files).load_data()
# #         index = VectorStoreIndex.from_documents(documents)
# #         query_engine = index.as_query_engine()
# #         return str(query_engine.query(message["text"]))
# #     else:  # If no files, just use the LLM directly
# #         return Settings.llm.complete(message["text"]).text  # Use the LLM to complete the prompt.
