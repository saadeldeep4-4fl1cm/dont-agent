FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=Qwen/Qwen2.5-Coder-1.5B \
    WORKSPACE_DIR=/tmp/agent_workspace \
    PORT=8000

WORKDIR /code
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY app ./app

EXPOSE 8000
# Single worker: the model is loaded once in-process. ${PORT} is set by Railway.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
