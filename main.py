"""
WhatsApp Brain — AI-powered RAG auto-reply bot for WhatsApp Business API
"""
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Brain", version="1.0.0")

rag = RAGEngine()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "whatsapp_brain_verify")


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()
    logger.info(f"Received: {body}")

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "no message"}

        message = value["messages"][0]
        from_number = message["from"]
        msg_type = message.get("type", "")

        if msg_type == "text":
            user_text = message["text"]["body"]
            logger.info(f"Message from {from_number}: {user_text}")

            # RAG-powered reply
            answer = rag.query(user_text)
            await send_whatsapp_message(from_number, answer)

    except (KeyError, IndexError) as e:
        logger.error(f"Parse error: {e}")

    return {"status": "ok"}


async def send_whatsapp_message(to: str, text: str):
    """Send a message via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        logger.info(f"Sent to {to}: {resp.status_code}")
        return resp.json()


class HistoryMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


@app.post("/ask")
async def ask(body: AskRequest):
    """Query the RAG engine directly — used by whatsapp-web.js bot."""
    history = [{"role": m.role, "content": m.content} for m in body.history]
    answer = rag.query(body.question, history=history)
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok", "docs_loaded": rag.doc_count()}
