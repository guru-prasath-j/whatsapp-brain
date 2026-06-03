# 🧠 WhatsApp Brain

> An AI-powered WhatsApp bot that reads your business documents and answers customer questions automatically — no manual replies needed.

Send it a WhatsApp message, it searches your knowledge base, and replies instantly using GPT-4o-mini. Drop in any PDF, text file, or markdown doc and it learns from it.

---

## How it works

```
Customer WhatsApp → Meta Cloud API → FastAPI Webhook → RAG Engine → GPT-4o-mini → Auto Reply
```

1. Customer sends a WhatsApp message
2. Meta forwards it to your webhook
3. The RAG engine searches your docs for relevant context
4. GPT-4o-mini generates a short, conversational reply
5. Reply is sent back via WhatsApp Cloud API

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/guru-prasath-j/whatsapp-brain.git
cd whatsapp-brain
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Fill in your WHATSAPP_TOKEN, PHONE_NUMBER_ID, and OPENAI_API_KEY
```

### 3. Add your documents
```bash
mkdir docs
# Drop in any .txt, .pdf, or .md files with your business info
```

### 4. Build the knowledge base
```bash
python ingest.py
```

### 5. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Expose via ngrok (for local testing)
```bash
ngrok http 8000
# Copy the https URL → paste into Meta webhook config as: https://xxxx.ngrok.io/webhook
```

### 7. Test locally (no WhatsApp needed)
```bash
python demo.py
```

---

## Meta Setup

1. Go to [developers.facebook.com](https://developers.facebook.com) → Create App → **Business Messaging**
2. Add WhatsApp product → **API Setup**
3. Copy **Temporary Access Token** and **Phone Number ID** → paste into `.env`
4. Under **Webhooks** → add your ngrok URL + verify token from `.env`
5. Subscribe to the **messages** field

---

## Project Structure

```
whatsapp-brain/
├── main.py          # FastAPI app + webhook handler
├── rag_engine.py    # Document loading, FAISS indexing, QA chain
├── ingest.py        # CLI: rebuild vector store from docs/
├── demo.py          # Test the bot locally
├── requirements.txt
├── .env.example
└── docs/            # ← drop your business docs here (gitignored)
```

---

## Tech Stack

| Layer | Tech |
|---|---|
| API Server | FastAPI + Uvicorn |
| Messaging | WhatsApp Cloud API (Meta) |
| RAG | LangChain + FAISS |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | OpenAI text-embedding-ada-002 |

---

Built by [Guruprasath J](https://github.com/guru-prasath-j)
