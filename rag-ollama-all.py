# # # # process_rag_chat.py (Example filename)
# # # from llama_index.core import (
# # #     VectorStoreIndex,
# # #     SimpleDirectoryReader,
# # #     Settings,
# # # )
# # # from llama_index.llms.ollama import Ollama
# # # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# # # import gradio as gr
# # # from ollama import Client
# # # import subprocess
# # # import logging
# # # import re
# # # import os

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # class TimeoutClient(Client):
# # #     def __init__(self, host="http://ollama:11434", timeout=600):  # 600 is 10 min. adjust as needed.
# # #         super().__init__(host=host)
# # #         self._client.timeout = timeout

# # # def get_running_ollama_model():
# # #     """
# # #     Retrieves the name of the currently running Ollama model using ps -ef.
# # #     """
# # #     try:
# # #         logging.info("Attempting to retrieve running Ollama model from process list...")
# # #         process = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# # #         stdout, stderr = process.communicate()
# # #         if stdout:
# # #             lines = stdout.decode().strip().split('\n')
# # #             for line in lines:
# # #                 if 'ollama run' in line:
# # #                     match = re.search(r'ollama run (\S+)', line)
# # #                     if match:
# # #                         model_name = match.group(1)
# # #                         logging.info(f"Detected running Ollama model: {model_name}")
# # #                         return model_name
# # #         logging.warning("No 'ollama run' process found.")
# # #         return None
# # #     except Exception as e:
# # #         logging.error(f"Error getting running Ollama model from process list: {e}")
# # #         return None

# # # # Get the running Ollama model
# # # running_model = get_running_ollama_model()

# # # # Configure Ollama LLM
# # # if running_model:
# # #     ollama_model = running_model
# # #     logging.info(f"Using running Ollama model: {ollama_model}")
# # # else:
# # #     ollama_model = "llama2:latest"  # Default if no running model is found.
# # #     logging.warning(f"No running Ollama model found. Using default: {ollama_model}")

# # # ollama_llm = Ollama(model=ollama_model, client=TimeoutClient())

# # # # Configure Embedding Model
# # # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# # # # Configure LlamaIndex Settings
# # # Settings.llm = ollama_llm
# # # Settings.embed_model = embed_model

# # # def answer(message, history):
# # #     files = []
# # #     if "files" in message and message["files"]:  # Check if files are uploaded
# # #         for msg in history:
# # #             if msg['role'] == "user" and isinstance(msg['content'], tuple):
# # #                 files.append(msg['content'][0])
# # #         for file in message["files"]:
# # #             files.append(file)

# # #         # Extend accepted file types
# # #         accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
# # #         filtered_files = [f for f in files if os.path.splitext(f.name)[1].lower() in accepted_files]

# # #         documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
# # #         index = VectorStoreIndex.from_documents(documents)
# # #         query_engine = index.as_query_engine()
# # #         return str(query_engine.query(message["text"]))
# # #     else:  # If no files, just use the LLM directly
# # #         return Settings.llm.complete(message["text"]).text  # Use the LLM to complete the prompt.

# # # demo = gr.ChatInterface(
# # #     answer,
# # #     type="messages",
# # #     title="Ollama Llama Index RAG Chatbot",
# # #     description="Upload any text, pdf, json, csv, markdown, html and docx files and ask questions about them, or chat without files.",
# # #     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
# # #     multimodal=True,
# # # )

# # # demo.launch(server_name="0.0.0.0", server_port=7860) # change to 0.0.0.0 for docker


# # # process_rag_chat.py
# # from llama_index.core import (
# #     VectorStoreIndex,
# #     SimpleDirectoryReader,
# #     Settings,
# # )
# # from llama_index.llms.ollama import Ollama
# # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# # import gradio as gr
# # from ollama import Client
# # import logging
# # import os
# # import requests

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # class TimeoutClient(Client):
# #     def __init__(self, host="http://ollama:11434", timeout=600):
# #         super().__init__(host=host)
# #         self._client.timeout = timeout

# # def get_running_ollama_model():
# #     """
# #     Retrieves the name of the currently running Ollama model using the Ollama API within a Docker container.
# #     """
# #     try:
# #         logging.info("Attempting to retrieve running Ollama model from Ollama API...")
# #         response = requests.get("http://ollama:11434/api/tags")
# #         response.raise_for_status()
# #         data = response.json()
# #         if "models" in data and data["models"]:
# #             model_names = [model["name"] for model in data["models"]]
# #             if model_names:
# #                 model_name = model_names[0]
# #                 logging.info(f"Detected running Ollama model: {model_name}")
# #                 return model_name
# #         logging.warning("No Ollama models found via API.")
# #         return None
# #     except requests.exceptions.RequestException as e:
# #         logging.error(f"Error getting running Ollama model from Ollama API: {e}")
# #         return None
# #     except Exception as e:
# #         logging.error(f"An unexpected error occurred: {e}")
# #         return None

# # # Get the running Ollama model
# # running_model = get_running_ollama_model()

# # # Configure Ollama LLM
# # if running_model:
# #     ollama_model = running_model
# #     logging.info(f"Using running Ollama model: {ollama_model}")
# # else:
# #     ollama_model = "llama2:latest"
# #     logging.warning(f"No running Ollama model found. Using default: {ollama_model}")

# # ollama_llm = Ollama(model=ollama_model, client=TimeoutClient())

# # # Configure Embedding Model
# # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# # # Configure LlamaIndex Settings
# # Settings.llm = ollama_llm
# # Settings.embed_model = embed_model

# # def answer(message, history):
# #     files = []
# #     if "files" in message and message["files"]:
# #         for msg in history:
# #             if msg['role'] == "user" and isinstance(msg['content'], tuple):
# #                 files.append(msg['content'][0])
# #         for file in message["files"]:
# #             files.append(file)

# #         accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
# #         filtered_files = [f for f in files if os.path.splitext(f.name)[1].lower() in accepted_files]

# #         documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
# #         index = VectorStoreIndex.from_documents(documents)
# #         query_engine = index.as_query_engine()
# #         return str(query_engine.query(message["text"]))
# #     else:
# #         return Settings.llm.complete(message["text"]).text

# # demo = gr.ChatInterface(
# #     answer,
# #     type="messages",
# #     title="Ollama Llama Index RAG Chatbot",
# #     description="Upload any text, pdf, json, csv, markdown, html and docx files and ask questions about them, or chat without files.",
# #     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
# #     multimodal=True,
# # )

# # demo.launch(server_name="0.0.0.0", server_port=7860)


# # process_rag_chat.py
# from llama_index.core import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     Settings,
# )
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import gradio as gr
# from ollama import Client
# import logging
# import os
# import requests

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class TimeoutClient(Client):
#     def __init__(self, host="http://ollama:11434", timeout=600):
#         super().__init__(host=host)
#         self._client.timeout = timeout

# def get_running_ollama_model():
#     """
#     Retrieves the name of the currently running Ollama model using the Ollama API.
#     """
#     try:
#         logging.info("Attempting to retrieve running Ollama model from Ollama API...")
#         response = requests.get("http://ollama:11434/api/tags")
#         response.raise_for_status()
#         data = response.json()
#         if "models" in data and data["models"]:
#             model_names = [model["name"] for model in data["models"]]
#             if model_names:
#                 model_name = model_names[0]
#                 logging.info(f"Detected running Ollama model: {model_name}")
#                 return model_name
#         logging.warning("No Ollama models found via API.")
#         return None
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error getting running Ollama model from Ollama API: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return None

# # Get the running Ollama model
# running_model = get_running_ollama_model()

# # Configure Ollama LLM
# if running_model:
#     ollama_model = running_model
#     logging.info(f"Using running Ollama model: {ollama_model}")
# else:
#     ollama_model = "llama2:latest"
#     logging.warning(f"No running Ollama model found. Using default: {ollama_model}")

# ollama_llm = Ollama(model=ollama_model, client=TimeoutClient())

# # Configure Embedding Model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# # Configure LlamaIndex Settings
# Settings.llm = ollama_llm
# Settings.embed_model = embed_model

# def answer(message, history):
#     files = []
#     if "files" in message and message["files"]:
#         for msg in history:
#             if msg['role'] == "user" and isinstance(msg['content'], tuple):
#                 files.append(msg['content'][0])
#         for file in message["files"]:
#             files.append(file)

#         accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
#         filtered_files = [f for f in files if os.path.splitext(f.name)[1].lower() in accepted_files]

#         documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
#         index = VectorStoreIndex.from_documents(documents)
#         query_engine = index.as_query_engine()
#         return str(query_engine.query(message["text"]))
#     else:
#         return Settings.llm.complete(message["text"]).text

# demo = gr.ChatInterface(
#     answer,
#     type="messages",
#     title="Ollama Llama Index RAG Chatbot",
#     description="Upload any text, pdf, json, csv, markdown, html and docx files and ask questions about them, or chat without files.",
#     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
#     multimodal=True,
# )

# demo.launch(server_name="0.0.0.0", server_port=7860)


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
    def __init__(self, host="http://ollama:11434", timeout=600):
        super().__init__(host=host)
        self._client.timeout = timeout

def get_running_ollama_model():
    """
    Retrieves the name of the currently running Ollama model using the Ollama API.
    """
    try:
        logging.info("Attempting to retrieve running Ollama model from Ollama API...")
        response = requests.get("http://ollama:11434/api/tags")
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
#     files = []
#     if "files" in message and message["files"]:
#         for msg in history:
#             if msg['role'] == "user" and isinstance(msg['content'], tuple):
#                 files.append(msg['content'][0])
#         for file in message["files"]:
#             files.append(file)

#         accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
#         filtered_files = [f for f in files if os.path.splitext(f.name)[1].lower() in accepted_files]

#         documents = SimpleDirectoryReader(input_files=filtered_files).load_data()
#         index = VectorStoreIndex.from_documents(documents)
#         query_engine = index.as_query_engine()
#         return str(query_engine.query(message["text"]))
#     else:
#         return Settings.llm.complete(message["text"]).text


# def answer(message, history):
#     files = []
#     if "files" in message and message["files"]:  # Check if new files are uploaded
#         for file in message["files"]:
#             files.append(file)

#     for msg in history:
#         if msg['role'] == "user":
#             if isinstance(msg['content'], tuple):
#                 # This is how Gradio represents previously uploaded files in history
#                 file_path = msg['content'][0]
#                 file_name = os.path.basename(file_path)
#                 files.append(gr.File(file_path, file_name))  # Create a Gradio File object
#             elif isinstance(msg['content'], list):
#                 # Handle multiple files uploaded in a single message (less common but possible)
#                 for item in msg['content']:
#                     if isinstance(item, tuple):
#                         file_path = item[0]
#                         file_name = os.path.basename(file_path)
#                         files.append(gr.File(file_path, file_name))

#     # Extend accepted file types
#     accepted_files = [".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]
#     filtered_files = [f for f in files if os.path.splitext(f.name)[1].lower() in accepted_files]

#     documents = SimpleDirectoryReader(input_files=[f.name for f in filtered_files]).load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     query_engine = index.as_query_engine()
#     return str(query_engine.query(message["text"]))


# demo = gr.ChatInterface(
#     answer,
#     type="messages",
#     title="Ollama Llama Index RAG Chatbot",
#     description="Upload any text, pdf, json, csv, markdown, html and docx files and ask questions about them, or chat without files.",
#     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
#     multimodal=True,
# )

# demo = gr.ChatInterface(
#     # answer,
#     type="messages",
#     title="Ollama Llama Index RAG Chatbot",
#     description="Upload any text or pdf files and ask questions about them, or chat without files.",
#     textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt"]),
#     multimodal=True,
# )

def answer(message, history):
    files = []
    if "files" in message and message["files"]:  # Check if files are uploaded
        for msg in history:
            if msg['role'] == "user" and isinstance(msg['content'], tuple):
                files.append(msg['content'][0])
        for file in message["files"]:
            files.append(file)

        documents = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return str(query_engine.query(message["text"]))
    else:  # If no files, just use the LLM directly
        return Settings.llm.complete(message["text"]).text  # Use the LLM to complete the prompt.

demo = gr.ChatInterface(
    answer,
    type="messages",
    title="Ollama Llama Index RAG Chatbot",
    description="Upload any text,pdf,json,html,docx,csv files and ask questions about them, or chat without files.",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt", ".json", ".csv", ".md", ".html", ".docx"]),
    multimodal=True,
)


demo.launch(server_name="0.0.0.0", server_port=7860)