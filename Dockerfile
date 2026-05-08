FROM python:3.11-slim
WORKDIR /opt/JuniorAGI
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY src/ ./src/
EXPOSE 8000
ENV PYTHONPATH="/opt/JuniorAGI/src"
CMD ["uvicorn", "api.node_server:app", "--host", "0.0.0.0", "--port", "8000"]
