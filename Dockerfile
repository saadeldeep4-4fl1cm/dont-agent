FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=Qwen/Qwen2.5-Coder-1.5B \
    ADAPTER_ID=prism-ta/dont-agent-qwen25-coder-1.5b-lora \
    WORKSPACE_DIR=/tmp/agent_workspace \
    ENABLE_CODE_EXEC=false \
    PORT=8000

WORKDIR /code
COPY requirements.txt .
# CPU-only torch (much smaller than the CUDA wheel) installed first so the
# torch>=2.2 line in requirements.txt is already satisfied.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt
COPY app ./app

# Pre-download base model + LoRA adapter at build time so the container
# starts fast and never re-downloads on restart (~3GB baked into the image).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-Coder-1.5B'); \
    snapshot_download('prism-ta/dont-agent-qwen25-coder-1.5b-lora')"

EXPOSE 8000
# Single worker: the model is loaded once in-process. ${PORT} is set by Railway.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
