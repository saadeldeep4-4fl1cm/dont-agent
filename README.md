# Don't Agent 🤖

An open-source programming AI agent: a fine-tuned **Qwen2.5-Coder-1.5B** model
(+ LoRA adapter, persona *"Don't Agent"*) wrapped in a ReAct-style tool-using
agent loop and served as a FastAPI service with a minimal chat UI.

## How it works

```
user message -> POST /chat -> ReAct loop (app/agent.py, max 6 steps)
                                 ├─ model thinks (app/model.py)
                                 ├─ may call tools (app/tools.py):
                                 │    run_python / read_file / write_file / list_dir
                                 └─ returns final answer
```

- `app/model.py` — loads the base model once; applies the LoRA adapter when
  `ADAPTER_ID` is set. 4-bit quantization on CUDA, float32 on CPU.
- `app/agent.py` — ReAct loop: the model emits `{"action": ..., "args": ...}`
  JSON to use tools, or `{"action": "final", "answer": ...}` to finish.
- `app/tools.py` — sandboxed tools (stdlib only), confined to `WORKSPACE_DIR`.
- `app/main.py` — FastAPI: `GET /` (chat UI), `GET /health`, `POST /chat`.

## Environment variables

| Variable         | Default                      | Description                                    |
|------------------|------------------------------|------------------------------------------------|
| `MODEL_ID`       | `Qwen/Qwen2.5-Coder-1.5B`    | Base model id or local path                    |
| `ADAPTER_ID`     | *(unset)*                    | LoRA adapter id/path (e.g. your HF repo)       |
| `WORKSPACE_DIR`  | `/tmp/agent_workspace`       | Sandbox dir for the agent's tools              |
| `MAX_NEW_TOKENS` | `512`                        | Max tokens per model generation                |
| `PORT`           | `8000`                       | HTTP port (Railway sets this automatically)    |

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

> ⚠️ **Honest note:** a 1.5B model on CPU is slow — expect tens of seconds per
> reply (each agent step runs a full generation). For real use, run it on a GPU
> (the service automatically uses 4-bit quantization when CUDA is available).
> Railway's default containers are CPU-only; attach a GPU volume/service if you
> need interactive speed.

## Deploy on Railway

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo** → select the repo.
3. Railway detects `railway.json` and builds the `Dockerfile`.
4. Optional: set `ADAPTER_ID` to your fine-tuned LoRA repo on Hugging Face
   (e.g. `yourname/dont-agent-lora`) under **Variables**.
5. Railway runs the `/health` healthcheck and gives you a public URL.

No model weights are stored in this repo — the model (and adapter) download
from Hugging Face on first startup.

## API

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "write a python fibonacci function and test it"}'
# -> {"reply": "...", "steps": 3}
```
