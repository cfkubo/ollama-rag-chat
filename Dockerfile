FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY process_rag_chat.py .

EXPOSE 7860

CMD ["python", "process_rag_chat.py"]