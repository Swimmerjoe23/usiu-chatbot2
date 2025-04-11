FROM python:3.10.13-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

# Fix protobuf issues with langchain & chromadb by using the pure-Python parser
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]