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
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
async def ask(body: AskRequest):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    profile = body.customer_profile.dict() if body.customer_profile else {}
    answer  = rag.query(body.question, history=history, customer_profile=profile)
    return {"answer": answer}


# ── /suggestions — 3 options ──────────────────────────────────────────────────
@app.post("/suggestions")
async def suggestions(body: AskRequest):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    profile = body.customer_profile.dict() if body.customer_profile else {}
    result  = rag.suggestions(body.question, history=history, customer_profile=profile)
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
    summary = rag.summarize_history(history)
    return {"summary": summary}


# ── /reload — re-index docs without restart ───────────────────────────────────
@app.post("/reload")
async def reload_docs():
    import threading
    def do_reload():
        logger.info("Reloading docs…")
        rag._build_from_docs()
        logger.info(f"Reload complete — {rag.doc_count()} vectors")
    threading.Thread(target=do_reload, daemon=True).start()
    return {"status": "reloading", "message": "Re-indexing started in background"}


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "docs_loaded": rag.doc_count(),
        "corrections": len(rag._corrections),
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
