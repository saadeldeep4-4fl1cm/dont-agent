"""FastAPI service for Don't Agent.

Routes:
    GET  /        minimal chat UI
    GET  /health  liveness probe (does NOT require the model to be loaded)
    POST /chat    {"message": str, "history": [...]} -> {"reply": str, "steps": int}

The model loads lazily on the first /chat call so the container starts fast
and passes Railway's healthcheck even while the model downloads.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .model import is_loaded

app = FastAPI(title="Don't Agent")

# If API_KEY is set, POST /chat requires it via the X-API-Key header.
# / and /health stay public (Railway's healthcheck needs /health).
API_KEY = os.environ.get("API_KEY", "").strip()


async def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    reply: str
    steps: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": is_loaded()}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    # generate() blocks; run it off the event loop.
    result = await asyncio.to_thread(run_agent, req.message, req.history)
    return ChatResponse(reply=result["reply"], steps=result["steps"])


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Don't Agent</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:0 auto;padding:16px;background:#0f1115;color:#e8e8e8}
#log{border:1px solid #333;border-radius:8px;padding:12px;height:60vh;overflow-y:auto;background:#16181d}
.msg{margin:8px 0;white-space:pre-wrap;word-break:break-word}
.user{color:#8ab4ff}.agent{color:#9fe8a9}.sys{color:#888;font-size:.85em}
#row{display:flex;gap:8px;margin-top:12px}
#inp{flex:1;padding:10px;border-radius:8px;border:1px solid #333;background:#16181d;color:#e8e8e8}
button{padding:10px 18px;border-radius:8px;border:0;background:#2f6fed;color:#fff;cursor:pointer}
button:disabled{opacity:.5}
</style>
</head>
<body>
<h2>🤖 Don't Agent</h2>
<div id="keyrow" style="display:flex;gap:8px;margin-bottom:8px">
<input id="key" type="password" placeholder="API key (if required)" autocomplete="off"
 style="flex:1;padding:8px;border-radius:8px;border:1px solid #333;background:#16181d;color:#e8e8e8">
</div>
<div id="log"></div>
<div id="row">
<input id="inp" placeholder="Ask me to write or debug code..." autocomplete="off">
<button id="btn" onclick="send()">Send</button>
</div>
<script>
const log=document.getElementById('log'),inp=document.getElementById('inp'),btn=document.getElementById('btn'),keyf=document.getElementById('key');
keyf.value=localStorage.getItem('dont_agent_key')||'';
keyf.addEventListener('change',()=>localStorage.setItem('dont_agent_key',keyf.value.trim()));
let history=[];
function add(cls,t){const d=document.createElement('div');d.className='msg '+cls;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
async function send(){
 const m=inp.value.trim();if(!m)return;inp.value='';btn.disabled=true;
 add('user','You: '+m);const th=add('sys','thinking…');
 try{
  const headers={'Content-Type':'application/json'};
  if(keyf.value.trim())headers['X-API-Key']=keyf.value.trim();
  const r=await fetch('/chat',{method:'POST',headers:headers,body:JSON.stringify({message:m,history:history})});
  if(!r.ok)throw new Error('HTTP '+r.status);
  const j=await r.json();th.remove();
  add('agent',"Don't Agent ("+j.steps+" steps): "+j.reply);
  history.push({role:'user',content:m},{role:'assistant',content:j.reply});
 }catch(e){th.remove();add('sys','error: '+e.message);}
 btn.disabled=false;inp.focus();
}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
add('sys',"I am Don't Agent, your programming AI agent. Ask me anything about code.");
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
