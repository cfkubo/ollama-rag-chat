FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag-ollama.py .

EXPOSE 7860

CMD ["python", "rag-ollama.py"]