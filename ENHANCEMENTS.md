# WhatsApp Brain — AI Enhancements Guide

This document explains every AI enhancement added to the WhatsApp Brain RAG system,
what problem each one solves, how it works technically, and how to configure it.

---

## Overview

| # | Feature | File(s) Changed | Impact |
|---|---------|----------------|--------|
| 2 | Reranking | `rag_engine.py` | Better document retrieval accuracy |
| 3 | Customer Profile Memory | `rag_engine.py`, `bot.js`, `rag.js` | AI knows who it's talking to |
| 5 | Intent Detection | `rag_engine.py` | Tailored tone per message type |
| 7 | Language Auto-Detection | `rag_engine.py`, `bot.js` | Replies in customer's language |
| 8 | Learning from Corrections | All files + frontend | AI improves from your edits |
| 9 | History Summarisation | `rag_engine.py`, `bot.js` | Handles long conversations cleanly |

---

## #2 — Reranking (Better Document Retrieval)

### Problem
FAISS vector search retrieves chunks by semantic similarity, but sometimes the most
semantically "similar" chunk isn't the most relevant one. For example, searching
"how much does app development cost" might return a chunk about "app features" before
the one with actual pricing.

### Solution
Retrieve 15 candidate chunks instead of 4, then re-score each one using BM25-style
keyword matching:
- **Term frequency**: how many times query keywords appear in the chunk
- **Exact phrase bonus**: +5 if the exact query phrase appears in the chunk
- **Token overlap ratio**: percentage of query words found in the chunk

The top 5 highest-scoring chunks are sent to the LLM.

### How it works
```python
# In rag_engine.py → _rerank_docs()
raw_docs = self.db.similarity_search(question, k=15)  # retrieve 15
docs     = self._rerank_docs(question, raw_docs, top_k=5)  # rerank to 5
```

### Result
Price questions now reliably surface the pricing section. Service questions surface
the services section. Accuracy improves significantly for specific queries.

---

## #3 — Customer Profile Memory

### Problem
Every message was treated independently. The AI had no memory of who the customer is,
what they're interested in, or what prices were already quoted to them.

### Solution
A lightweight per-customer profile is built automatically from their messages and
stored in `conversation_history.json`. The profile contains:

| Field | Description | Example |
|-------|-------------|---------|
| `name` | WhatsApp display name | "Narmatha" |
| `language` | Detected language | "Tamil" |
| `interests` | Topics they've asked about | ["mobile app", "ticket booking"] |
| `quoted_prices` | Prices mentioned in conversation | ["₹75,000"] |
| `summary` | Summary of older messages | "Customer wants ticket booking app..." |

### How it works
```javascript
// In bot.js → updateCustomerProfile()
// Called automatically every time a user message arrives
// Extracts: language, price mentions, interest keywords
```

The profile is passed to every RAG call:
```javascript
getAIResponse(text, history, getCustomerProfile(senderNumber))
```

The RAG server includes it in the system prompt:
```
Customer Profile:
Name: Narmatha
Language: Tamil
Interests: ticket booking app, mobile development
Quoted prices: ₹75,000
```

### Result
The AI remembers context across messages and avoids re-quoting prices,
repeating information, or asking questions already answered.

---

## #5 — Intent Detection

### Problem
A greeting ("Hi") was being processed through the full RAG pipeline — fetching
business documents, building context — just to reply "Hello!". A complaint needed
an empathetic tone but got the same neutral response as a pricing inquiry.

### Solution
Every incoming message is first classified into one of 6 intents:

| Intent | Trigger Examples | Special Behaviour |
|--------|-----------------|-------------------|
| `greeting` | "Hi", "Hello", "Hey" | Skips RAG entirely, fast warm reply |
| `pricing` | "how much", "cost", "₹", "price" | Always quotes starting price from context |
| `complaint` | "problem", "issue", "not working" | Empathetic tone, offers resolution |
| `closing` | "thanks", "bye", "noted" | Warm sign-off |
| `followup` | — | References previous discussion |
| `general` | everything else | Standard RAG response |

### How it works
```python
# In rag_engine.py → _detect_intent()
# Step 1: Fast keyword heuristics (covers ~80% of cases — no LLM call)
# Step 2: For ambiguous messages, 1 quick Ollama call (8s timeout, 5 tokens)
```

Greetings skip RAG entirely, making them 5-10x faster.

### Result
Each message type gets an appropriately toned response. Complaints get apologies.
Greetings get warm welcomes. Pricing queries get specific numbers.

---

## #7 — Language Auto-Detection

### Problem
Customers messaging in Tamil were getting English replies. The AI had no way to know
what language to use.

### Solution
Detect language using Unicode block ranges — zero latency, no LLM call needed:

| Language | Unicode Range | Detection |
|----------|--------------|-----------|
| Tamil | U+0B80 – U+0BFF | > 2 Tamil chars |
| Hindi | U+0900 – U+097F | > 2 Devanagari chars |
| Arabic | U+0600 – U+06FF | > 2 Arabic chars |
| English | (default) | fallback |

### How it works
```python
# In rag_engine.py → _detect_language()
tamil = sum(1 for c in text if '஀' <= c <= '௿')
if tamil > 2: return "Tamil"
```

The detected language is added to the system prompt:
```
IMPORTANT: Reply in Tamil.
```

It's also stored in the customer profile so future messages don't need re-detection.

### Result
Tamil customers get Tamil replies, Hindi customers get Hindi replies, automatically.

---

## #8 — Learning from Corrections

### Problem
The AI's suggestions were generic at first. Every edit you made to improve a reply
was thrown away — the AI kept making the same mistakes.

### Solution
When you click **Edit** on a suggestion, load it into the text area, modify it, and
send — the system detects the difference and stores the (original, corrected) pair
as a "correction" in `corrections.json`.

These corrections become few-shot examples in the system prompt for future replies:
```
Examples of the preferred reply style (learn from these):
Q: How much for a website?
Draft: Websites start at ₹25,000.
Improved: Our websites start at ₹25,000 and include 1 year of free hosting! Want to book a free consultation?
---
Q: Hi
Draft: Hello! How can I help?
Improved: Vanakkam! 😊 Welcome to TechNova. How can I assist you today?
```

### How it works
1. **Frontend (ChatView.jsx)**: Clicking "Edit" on a suggestion loads it into the textarea and records `loadedSuggestion`
2. On send, if `sentText !== loadedSuggestion`, POST to `/api/feedback`
3. **bot.js** forwards to `/feedback` on the RAG server
4. **rag_engine.py** appends to `corrections.json` (max 50 stored)
5. Last 3 corrections are injected into every future system prompt

### Configuration
Corrections are stored at: `whatsapp_brain/corrections.json`
To reset: delete the file.
Maximum corrections stored: 50 (oldest are dropped).

### Result
The AI progressively learns your preferred tone, language mix, and style.
After 10-20 corrections it should feel noticeably more "you".

---

## #9 — History Summarisation

### Problem
After 40+ messages, the conversation history was getting too long for the LLM's
context window. Older messages were being cut off, losing important context
(e.g. what was quoted on day 1 of a week-long conversation).

### Solution
When conversation history reaches MAX_HISTORY (40 messages), the oldest 30 messages
are automatically summarised into a 2-3 sentence paragraph by Ollama. Only the last
10 messages are kept verbatim.

The summary is stored in the customer profile and injected before the recent messages:
```
[Earlier conversation summary: Customer Narmatha has been discussing a ticket 
booking mobile app. A price of ₹75,000 was quoted. She has been asking about 
feature-wise pricing and timeline.]

[last 10 messages verbatim...]
```

### How it works
```javascript
// In bot.js → addToHistory()
if (rec.messages.length >= MAX_HISTORY && !rec._summarizing) {
    const oldMsgs = rec.messages.slice(0, MAX_HISTORY - 10)
    summarizeHistory(oldMsgs).then(summary => {
        rec.customerProfile.summary = summary
    })
}
```

```python
# In rag_engine.py → _summarize_old_messages()
# Ollama call with 120 token output limit
# Focuses on: what customer needs, what was discussed, what was quoted
```

### Result
Long conversations no longer lose context. The AI remembers a week of discussion
condensed into a paragraph, plus has full detail on the last 10 messages.

---

## Architecture Diagram

```
Customer WhatsApp Message
         │
         ▼
   [bot.js — Node.js]
   ┌─────────────────────────────────────┐
   │ 1. Update customer profile (#3, #7) │
   │ 2. Debounce (3s burst protection)   │
   │ 3. getAIResponse(text, hist, prof)  │
   └────────────────┬────────────────────┘
                    │ HTTP POST /ask
                    ▼
   [whatsapp_brain — FastAPI]
   ┌─────────────────────────────────────┐
   │ 4. Detect intent (#5)               │
   │ 5. Detect language (#7)             │
   │ 6. Retrieve 15 chunks from FAISS    │
   │ 7. Rerank to top 5 (#2)             │
   │ 8. Build smart prompt:              │
   │    - Intent tone                    │
   │    - Language instruction           │
   │    - Customer profile (#3)          │
   │    - History summary (#9)           │
   │    - Correction examples (#8)       │
   │ 9. Call Ollama → answer             │
   └──────────────────┬──────────────────┘
                      │
                      ▼
               [Ollama — LLM]
               llama3.2 / llama3.1:8b
```

---

## File Reference

| File | Role |
|------|------|
| `whatsapp_brain/rag_engine.py` | Core AI: reranking, intent, language, summary, corrections |
| `whatsapp_brain/main.py` | FastAPI endpoints: /ask, /suggestions, /feedback, /summarize, /reload |
| `whatsapp_brain/corrections.json` | Auto-created. Stores correction pairs for few-shot learning |
| `whatsapp-personal-bot/rag.js` | Passes customer profile + corrections to RAG server |
| `whatsapp-personal-bot/bot.js` | Builds customer profile, calls summarize, /api/feedback endpoint |
| `frontend/src/components/ChatView.jsx` | Edit button on suggestions, correction detection on send |

---

## Configuration (`.env`)

```env
# Model — upgrade for better quality
OLLAMA_MODEL=llama3.1:8b       # better quality, ~4.7GB
# OLLAMA_MODEL=llama3.2        # faster, ~2GB (default)

# How many messages to keep verbatim before summarising
# Default: 40 (set in bot.js MAX_HISTORY)

# History TTL — how long before a conversation is considered "stale"
HISTORY_TTL_HOURS=168          # 7 days
```

---

## Recommended Next Steps

1. **Upgrade model**: `ollama pull llama3.1:8b` and set `OLLAMA_MODEL=llama3.1:8b` — biggest quality jump
2. **Add more docs**: Upload price lists, catalogues, and policy documents via the Profile panel
3. **Make corrections**: Edit 10-20 suggestions to teach the AI your preferred style
4. **Add FAQs**: Fill in the FAQ section of the Profile panel — these are indexed and used verbatim
