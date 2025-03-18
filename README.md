#### Demo application to showcase RAG with Ollama models

#### Requiremnts
- Python 3.13.x
- Docker
- hugginface account
- ollama models

#### Running it locally

1. Clone the repository
```
git clone https://github.com/ollama/rag-ollama.git
```

2. Create a virtual environment and activate it
```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies using pip
```
pip install -r requirements.txt
```

4. Run ollama
```
ollama run llama2:latest
```

. Run the application
```
python3 rag-ollama-all-local.py
```


```
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
```

docker system prune -a --volumes

docker volume rm $(docker volume ls -q)