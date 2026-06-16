"""
WhatsApp Brain — Enhanced RAG API
Endpoints:
  POST /ask         — single auto-reply
  POST /suggestions — 3 reply options
  POST /feedback    — store correction pair (#8)
  POST /summarize   — summarise old history (#9)
  POST /reload      — re-index docs without restart
  GET  /health      — status check
  GET  /webhook     — Meta webhook verify
  POST /webhook     — Meta webhook receive
"""
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # must run before rag_engine import so DOCS_DIR env var is set

from fastapi import BackgroundTasks, FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
from rag_engine import RAGEngine, DOCS_DIR, OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_MODEL
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUSINESS_INFO_FILE = os.path.join(DOCS_DIR, "business_info.txt")

app = FastAPI(title="WhatsApp Brain", version="2.0.0")
rag = RAGEngine()

WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "whatsapp_brain_verify")


# ── Shared models ─────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role:    str
    content: str

class CustomerProfile(BaseModel):
    name:          Optional[str] = None
    language:      Optional[str] = None
    interests:     Optional[List[str]] = []
    quoted_prices: Optional[List[str]] = []
    summary:       Optional[str] = None

class AskRequest(BaseModel):
    question:         str
    history:          List[HistoryMessage] = []
    customer_profile: Optional[CustomerProfile] = None


# ── /ask — single reply ───────────────────────────────────────────────────────
@app.post("/ask")
async def ask(body: AskRequest, background_tasks: BackgroundTasks):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    profile = body.customer_profile.dict() if body.customer_profile else {}

    # Determine which messages are "old" (same split as _build_messages)
    prior    = [m for m in history if not (m["role"] == "user" and m["content"] == body.question)]
    old_msgs = prior[:-10] if len(prior) > 10 else []

    new_summary = None
    if old_msgs and not profile.get("summary"):
        cache_key = rag._summary_cache_key(old_msgs)
        with rag._summary_lock:
            cached = rag._summary_cache.get(cache_key)
        if cached:
            # Cache hit — inject so this request benefits too
            profile["summary"] = cached
            new_summary = cached
        else:
            # Schedule summarization to run after response is sent (non-blocking)
            background_tasks.add_task(rag._run_summary_background, old_msgs, cache_key)

    answer = await asyncio.to_thread(rag.query, body.question, history, profile)
    response: dict = {"answer": answer}
    if new_summary:
        response["new_summary"] = new_summary
    return response


# ── /suggestions — 3 options ──────────────────────────────────────────────────
@app.post("/suggestions")
async def suggestions(body: AskRequest):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    profile = body.customer_profile.dict() if body.customer_profile else {}
    result  = await asyncio.to_thread(rag.suggestions, body.question, history, profile)
    return {"suggestions": result}


# ── /feedback — store correction pair (#8) ────────────────────────────────────
class FeedbackRequest(BaseModel):
    question:  str
    original:  str
    corrected: str

@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    """Store a human-edited suggestion as a few-shot example for future replies."""
    rag.add_correction(body.question, body.original, body.corrected)
    return {"ok": True, "total_corrections": len(rag._corrections)}


# ── /summarize — summarise old history (#9) ───────────────────────────────────
class SummarizeRequest(BaseModel):
    history: List[HistoryMessage]

@app.post("/summarize")
async def summarize(body: SummarizeRequest):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    summary = await asyncio.to_thread(rag.summarize_history, history)
    return {"summary": summary}


# ── /reload — re-index docs without restart ───────────────────────────────────
@app.post("/switch-model")
async def switch_model(request: Request):
    """Hot-swap the Ollama model without restarting."""
    body = await request.json()
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Missing model")
    import rag_engine as re_module
    re_module.OLLAMA_MODEL = model
    rag._corrections = rag._load_corrections()  # reload in case corrections reference old model
    logger.info(f"Switched model to: {model}")
    return {"ok": True, "model": model}

@app.post("/reload")
async def reload_docs():
    import threading
    def do_reload():
        logger.info(f"[Reload] Starting — DOCS_DIR={DOCS_DIR}")
        old_db = rag.db
        try:
            rag._build_from_docs()
        except Exception as e:
            logger.error(f"[Reload] _build_from_docs threw: {e} — keeping old index")
            rag.db = old_db
            return
        if rag.db is None and old_db is not None:
            logger.warning("[Reload] No docs found after reload — keeping old index to avoid outage")
            rag.db = old_db
        logger.info(f"[Reload] Done — {rag.doc_count()} vectors")
    threading.Thread(target=do_reload, daemon=True).start()
    return {"status": "reloading", "message": "Re-indexing started in background"}


# ── /docs-content — read / write business info ───────────────────────────────
@app.get("/docs-content")
async def get_docs_content():
    os.makedirs(DOCS_DIR, exist_ok=True)
    if not os.path.exists(BUSINESS_INFO_FILE):
        return {"content": "", "vectors": rag.doc_count()}
    content = Path(BUSINESS_INFO_FILE).read_text(encoding="utf-8")
    return {"content": content, "vectors": rag.doc_count()}


class DocsContentRequest(BaseModel):
    content: str

@app.post("/docs-content")
async def save_docs_content(body: DocsContentRequest):
    os.makedirs(DOCS_DIR, exist_ok=True)
    Path(BUSINESS_INFO_FILE).write_text(body.content.strip(), encoding="utf-8")
    # Trigger rebuild in background
    import threading
    def do_reload():
        old_db = rag.db
        try:
            rag._build_from_docs()
        except Exception as e:
            logger.error(f"[Reload] {e}")
            rag.db = old_db
        logger.info(f"[Reload] Done — {rag.doc_count()} vectors")
    threading.Thread(target=do_reload, daemon=True).start()
    return {"ok": True, "message": "Saved and re-indexing in background"}


# ── / — chat + docs editor UI ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=_CHAT_UI_HTML)


_CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhatsApp Brain</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  body { display: flex; height: 100vh; background: #f0f2f5; }

  /* ── Left panel — docs editor ── */
  #docs-panel {
    width: 380px; min-width: 280px; display: flex; flex-direction: column;
    background: #fff; border-right: 1px solid #ddd;
  }
  #docs-panel header {
    background: #075e54; color: #fff; padding: 14px 16px;
    font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 10px;
  }
  #docs-panel header span { font-size: 20px; }
  #docs-status {
    padding: 8px 14px; font-size: 12px; color: #666; background: #f9f9f9;
    border-bottom: 1px solid #eee;
  }
  #docs-status b { color: #075e54; }
  #business-info {
    flex: 1; resize: none; border: none; outline: none; padding: 14px;
    font-size: 13px; line-height: 1.6; color: #333;
  }
  #docs-actions {
    padding: 10px 14px; border-top: 1px solid #eee; display: flex; gap: 8px;
  }
  #save-btn {
    flex: 1; background: #075e54; color: #fff; border: none; border-radius: 6px;
    padding: 10px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  #save-btn:hover { background: #054c44; }
  #save-btn:disabled { background: #aaa; cursor: default; }
  #save-status { font-size: 12px; color: #666; align-self: center; }

  /* ── Right panel — chat ── */
  #chat-panel { flex: 1; display: flex; flex-direction: column; }
  #chat-panel header {
    background: #075e54; color: #fff; padding: 14px 18px;
    font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 10px;
  }
  #chat-panel header span { font-size: 20px; }
  #messages {
    flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px;
    background: #e5ddd5 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3C/svg%3E");
  }
  .bubble {
    max-width: 72%; padding: 9px 13px; border-radius: 8px;
    font-size: 14px; line-height: 1.5; word-break: break-word; position: relative;
  }
  .bubble.user { background: #dcf8c6; align-self: flex-end; border-bottom-right-radius: 2px; }
  .bubble.bot  { background: #fff; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,.1); }
  .bubble.typing { color: #999; font-style: italic; }
  #input-bar {
    display: flex; gap: 8px; padding: 10px 14px;
    background: #f0f2f5; border-top: 1px solid #ddd; align-items: center;
  }
  #question {
    flex: 1; border: none; border-radius: 20px; padding: 10px 16px;
    font-size: 14px; outline: none; background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,.1);
  }
  #send-btn {
    background: #075e54; color: #fff; border: none; border-radius: 50%;
    width: 44px; height: 44px; font-size: 18px; cursor: pointer; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
  }
  #send-btn:hover { background: #054c44; }
  #send-btn:disabled { background: #aaa; }
</style>
</head>
<body>

<!-- Left: Docs Editor -->
<div id="docs-panel">
  <header><span>📄</span> Business Info</header>
  <div id="docs-status">Loading…</div>
  <textarea id="business-info" placeholder="Paste your business information here.

Examples:
- Business name and description
- Services or products offered with prices
- Working hours and location
- Contact details
- FAQs

The bot will answer customer questions using only what you write here."></textarea>
  <div id="docs-actions">
    <button id="save-btn" onclick="saveDocs()">Save &amp; Reload Bot</button>
    <span id="save-status"></span>
  </div>
</div>

<!-- Right: Chat -->
<div id="chat-panel">
  <header><span>🤖</span> Test Chat</header>
  <div id="messages">
    <div class="bubble bot">Hi! Ask me anything about the business. Make sure you save your business info first 👈</div>
  </div>
  <div id="input-bar">
    <input id="question" type="text" placeholder="Type a question…" onkeydown="if(event.key==='Enter') sendMsg()">
    <button id="send-btn" onclick="sendMsg()">➤</button>
  </div>
</div>

<script>
const history = [];

async function loadDocs() {
  try {
    const r = await fetch('/docs-content');
    const d = await r.json();
    document.getElementById('business-info').value = d.content || '';
    document.getElementById('docs-status').innerHTML =
      d.vectors > 0
        ? `<b>${d.vectors} vectors</b> indexed — bot is ready`
        : `<b>No vectors</b> — add business info and save`;
  } catch(e) {
    document.getElementById('docs-status').textContent = 'Could not load docs info';
  }
}

async function saveDocs() {
  const btn = document.getElementById('save-btn');
  const status = document.getElementById('save-status');
  const content = document.getElementById('business-info').value.trim();
  if (!content) { status.textContent = 'Nothing to save.'; return; }
  btn.disabled = true;
  btn.textContent = 'Saving…';
  status.textContent = '';
  try {
    await fetch('/docs-content', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ content })
    });
    btn.textContent = 'Save & Reload Bot';
    status.textContent = '✓ Saved! Re-indexing…';
    // Poll until vectors change
    const before = parseInt(document.getElementById('docs-status').textContent) || 0;
    let attempts = 0;
    const poll = setInterval(async () => {
      attempts++;
      const r = await fetch('/docs-content');
      const d = await r.json();
      if (d.vectors > 0 || attempts > 15) {
        clearInterval(poll);
        document.getElementById('docs-status').innerHTML =
          `<b>${d.vectors} vectors</b> indexed — bot is ready`;
        status.textContent = '✓ Ready!';
        btn.disabled = false;
      }
    }, 2000);
  } catch(e) {
    status.textContent = '✗ Error saving.';
    btn.disabled = false;
    btn.textContent = 'Save & Reload Bot';
  }
}

async function sendMsg() {
  const input = document.getElementById('question');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';

  addBubble(q, 'user');
  history.push({ role: 'user', content: q });

  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  const typing = addBubble('typing…', 'bot typing');

  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question: q, history: history.slice(-20) })
    });
    const d = await r.json();
    typing.remove();
    const answer = d.answer || 'No response.';
    addBubble(answer, 'bot');
    history.push({ role: 'assistant', content: answer });
  } catch(e) {
    typing.remove();
    addBubble('Error — is the server running?', 'bot');
  }
  btn.disabled = false;
  input.focus();
}

function addBubble(text, cls) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'bubble ' + cls;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

loadDocs();
</script>
</body>
</html>"""


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if OPENAI_API_KEY:
        provider = f"openai/{OPENAI_MODEL}"
    else:
        provider = f"ollama/{OLLAMA_MODEL}"
    return {
        "status":       "ok",
        "llm_provider": provider,
        "docs_loaded":  rag.doc_count(),
        "corrections":  len(rag._corrections),
    }


# ── Meta webhook (legacy) ─────────────────────────────────────────────────────
@app.get("/webhook")
async def verify_webhook(request: Request):
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    try:
        message     = body["entry"][0]["changes"][0]["value"]["messages"][0]
        from_number = message["from"]
        if message.get("type") == "text":
            user_text = message["text"]["body"]
            answer    = rag.query(user_text)
            await _send_whatsapp_message(from_number, answer)
    except (KeyError, IndexError) as e:
        logger.error(f"Parse error: {e}")
    return {"status": "ok"}


async def _send_whatsapp_message(to: str, text: str):
    url     = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        logger.info(f"Sent to {to}: {resp.status_code}")
