"""
RAG Engine — Enhanced with:
  #2  Reranking        : retrieve 15 chunks, BM25-score, return top 5
  #3  Customer Profile : per-customer memory (name, language, interests, quotes)
  #5  Intent Detection : route greeting/pricing/complaint/followup/closing/general
  #7  Language Detection: reply in Tamil / Hindi / English automatically
  #8  Correction Learning: few-shot examples from human-edited suggestions
  #9  History Summary  : summarise messages older than 20 to save context window
"""

import os
import re
import glob
import json
import logging
import httpx
from typing import Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR           = os.getenv("DOCS_DIR",           "docs")
VECTOR_STORE_PATH  = os.getenv("VECTOR_STORE_PATH",  "vector_store")
CORRECTIONS_FILE   = os.getenv("CORRECTIONS_FILE",   "corrections.json")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

INTENT_TYPES = {"greeting", "pricing", "complaint", "followup", "closing", "general"}

SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant for a business. "
    "Answer using ONLY the business context provided. "
    "If the answer is not in the context say: "
    "'I don't have that information right now. Please contact us directly.' "
    "Keep replies short and conversational (max 3 sentences). No markdown."
)


# ── RAG Engine ────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        self.db: Optional[FAISS] = None
        self._corrections: list  = self._load_corrections()
        self._load()

    # ── Vector store ─────────────────────────────────────────────────────────
    def _embeddings(self):
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def _load(self):
        os.makedirs(DOCS_DIR, exist_ok=True)
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                self.db = FAISS.load_local(
                    VECTOR_STORE_PATH, self._embeddings(),
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded vector store — {self.doc_count()} vectors")
                return
            except Exception as e:
                logger.warning(f"Could not load store: {e}. Rebuilding…")
        self._build_from_docs()

    def _build_from_docs(self):
        docs = []
        for pattern in [f"{DOCS_DIR}/**/*.txt", f"{DOCS_DIR}/**/*.md"]:
            for f in glob.glob(pattern, recursive=True):
                try:   docs.extend(TextLoader(f, encoding="utf-8").load())
                except Exception as e: logger.warning(f"Skipping {f}: {e}")
        for f in glob.glob(f"{DOCS_DIR}/**/*.pdf", recursive=True):
            try:   docs.extend(PyPDFLoader(f).load())
            except Exception as e: logger.warning(f"Skipping {f}: {e}")

        if not docs:
            logger.warning("No documents in docs/. Using fallback mode.")
            self.db = None
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_documents(docs)
        logger.info(f"Indexed {len(chunks)} chunks from {len(docs)} documents")
        self.db = FAISS.from_documents(chunks, self._embeddings())
        self.db.save_local(VECTOR_STORE_PATH)

    # ── #2 Reranking ─────────────────────────────────────────────────────────
    def _rerank_docs(self, question: str, docs: list, top_k: int = 5) -> list:
        """
        Retrieve more chunks than needed, then score each by BM25-style term
        overlap with the question.  Returns the top_k highest-scoring chunks.
        This improves relevance compared to pure vector similarity alone.
        """
        if len(docs) <= top_k:
            return docs

        q_terms = set(re.findall(r'\w+', question.lower()))
        scored  = []
        for i, doc in enumerate(docs):
            content = doc.page_content.lower()
            words   = set(re.findall(r'\w+', content))

            # Term frequency: how many query terms appear in the chunk
            tf_score = sum(content.count(t) for t in q_terms)

            # Exact phrase bonus
            phrase_bonus = 5 if question.lower()[:50] in content else 0

            # Token overlap ratio
            overlap = len(q_terms & words) / max(len(q_terms), 1)

            scored.append((tf_score + phrase_bonus + overlap * 3, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    # ── #7 Language Detection ────────────────────────────────────────────────
    def _detect_language(self, text: str) -> str:
        """
        Detect language using Unicode block ranges — no external library needed.
        Tamil  : U+0B80–U+0BFF
        Hindi  : U+0900–U+097F
        Arabic : U+0600–U+06FF
        Falls back to English.
        """
        tamil  = sum(1 for c in text if '஀' <= c <= '௿')
        hindi  = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
        arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')

        if tamil  > 2: return "Tamil"
        if hindi  > 2: return "Hindi"
        if arabic > 2: return "Arabic"
        return "English"

    # ── #5 Intent Detection ──────────────────────────────────────────────────
    def _detect_intent(self, question: str) -> str:
        """
        Classify the message intent so the reply prompt can be tailored.
        Uses fast keyword heuristics first; falls back to a tiny Ollama call
        only for ambiguous messages.
        """
        q = question.lower().strip()

        # Fast heuristics — covers ~80 % of cases without an LLM call
        greeting_words = {"hi", "hello", "hey", "hai", "hii", "vanakkam",
                          "good morning", "good evening", "good afternoon", "sup"}
        if q in greeting_words or (len(q.split()) <= 3 and any(g in q for g in greeting_words)):
            return "greeting"

        if any(w in q for w in ["price", "cost", "how much", "charges", "fee",
                                 "rate", "₹", "rs.", "rupee", "budget", "quote",
                                 "estimate", "package"]):
            return "pricing"

        if any(w in q for w in ["problem", "issue", "not working", "broken",
                                 "complaint", "bad", "worst", "disappointed",
                                 "refund", "cancel", "wrong"]):
            return "complaint"

        if any(w in q for w in ["thanks", "thank you", "ok", "okay", "noted",
                                 "got it", "sure", "bye", "goodbye", "see you",
                                 "will do", "fine"]):
            return "closing"

        # LLM fallback for ambiguous messages
        try:
            with httpx.Client(timeout=8.0) as client:
                r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                    "model": OLLAMA_MODEL,
                    "prompt": (
                        "Classify this WhatsApp message into ONE word from: "
                        "greeting, pricing, complaint, followup, closing, general\n"
                        f"Message: {question[:200]}\nAnswer:"
                    ),
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 5},
                })
            intent = r.json().get("response", "general").strip().lower()
            if intent in INTENT_TYPES:
                return intent
        except Exception:
            pass

        return "general"

    # ── #9 History Summarisation ─────────────────────────────────────────────
    def _summarize_old_messages(self, old_messages: list) -> str:
        """
        Condense messages older than the recent window into a single paragraph.
        This frees up context-window space while preserving key facts.
        """
        if not old_messages:
            return ""
        text_block = "\n".join(
            f"{'Customer' if m['role']=='user' else 'Agent'}: {m['content']}"
            for m in old_messages
        )
        prompt = (
            "Summarise this WhatsApp conversation in 2-3 sentences, "
            "focusing on what the customer needs and what was discussed:\n\n"
            f"{text_block}\n\nSummary:"
        )
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 120},
                })
            return r.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Summary failed: {e}")
            return ""

    # ── #8 Corrections ───────────────────────────────────────────────────────
    def _load_corrections(self) -> list:
        if os.path.exists(CORRECTIONS_FILE):
            try:
                with open(CORRECTIONS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def add_correction(self, question: str, original: str, corrected: str):
        """Store a human-edited suggestion as a few-shot example."""
        self._corrections.append({
            "question":  question,
            "original":  original,
            "corrected": corrected,
        })
        # Keep last 50 corrections
        self._corrections = self._corrections[-50:]
        try:
            with open(CORRECTIONS_FILE, "w") as f:
                json.dump(self._corrections, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save corrections: {e}")

    # ── Smart prompt builder ─────────────────────────────────────────────────
    def _build_messages(
        self,
        question:         str,
        context:          str,
        intent:           str,
        language:         str,
        history:          list,
        customer_profile: dict,
        for_suggestions:  bool = False,
    ) -> list:
        """
        Assemble the full message list for Ollama, incorporating:
        - intent-specific tone instruction
        - language instruction
        - customer profile context
        - history summary (#9)
        - few-shot corrections (#8)
        """

        # Intent tone instructions
        tone_map = {
            "greeting":  "This is a greeting — be warm, welcoming, and briefly mention 1-2 key services.",
            "pricing":   "Customer is asking about pricing — be specific, always quote the starting price from context.",
            "complaint": "Customer seems unhappy — be empathetic and apologetic first, then offer a clear resolution.",
            "closing":   "Conversation is wrapping up — give a warm sign-off and invite future contact.",
            "followup":  "This is a follow-up — reference what was discussed before if visible in history.",
            "general":   "",
        }
        tone = tone_map.get(intent, "")

        # Language instruction
        lang_note = f"IMPORTANT: Reply in {language}." if language != "English" else ""

        # Corrections few-shot (#8)
        correction_block = ""
        if self._corrections:
            examples = self._corrections[-3:]
            lines = [
                f"Q: {c['question']}\nDraft: {c['original']}\nImproved: {c['corrected']}"
                for c in examples
            ]
            correction_block = (
                "\n\nExamples of the preferred reply style (learn from these):\n"
                + "\n---\n".join(lines)
            )

        # Customer profile (#3)
        profile_block = ""
        if customer_profile:
            parts = []
            if customer_profile.get("name"):
                parts.append(f"Name: {customer_profile['name']}")
            if customer_profile.get("language"):
                parts.append(f"Language: {customer_profile['language']}")
            if customer_profile.get("interests"):
                parts.append(f"Interests: {', '.join(customer_profile['interests'][-5:])}")
            if customer_profile.get("quoted_prices"):
                parts.append(f"Quoted prices: {', '.join(customer_profile['quoted_prices'][-3:])}")
            if parts:
                profile_block = "\n\nCustomer Profile:\n" + "\n".join(parts)

        # Build system prompt
        suggestion_note = (
            "\n\nGenerate exactly 3 reply options as a JSON array: [\"reply1\",\"reply2\",\"reply3\"]\n"
            "Vary tone: 1=concise, 2=warm, 3=detailed. No markdown."
        ) if for_suggestions else ""

        system = (
            SYSTEM_PROMPT
            + (f"\n{tone}" if tone else "")
            + (f"\n{lang_note}" if lang_note else "")
            + profile_block
            + correction_block
            + suggestion_note
        )

        # History with summary (#9)
        # Split: summarise old messages, keep last 10 verbatim
        prior = [m for m in history if not (m["role"] == "user" and m["content"] == question)]
        old_msgs    = prior[:-10] if len(prior) > 10 else []
        recent_msgs = prior[-10:]

        messages = [{"role": "system", "content": system}]

        # Inject summary if we have old messages
        summary = customer_profile.get("summary", "") if customer_profile else ""
        if not summary and old_msgs:
            summary = self._summarize_old_messages(old_msgs)

        if summary:
            messages.append({
                "role":    "system",
                "content": f"[Earlier conversation summary: {summary}]"
            })

        messages.extend(recent_msgs)
        messages.append({
            "role":    "user",
            "content": f"Business Context:\n{context}\n\nCustomer message: {question}",
        })

        return messages

    # ── Ollama call ───────────────────────────────────────────────────────────
    def _ollama(self, messages: list, temperature: float = 0.2,
                json_format: bool = False) -> str:
        payload = {
            "model":    OLLAMA_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature},
        }
        if json_format:
            payload["format"] = "json"
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except httpx.ConnectError:
            return "Ollama is not running. Please start it with: ollama serve"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    # ── Public: single answer ─────────────────────────────────────────────────
    def query(self, question: str, history: list = [],
              customer_profile: dict = {}) -> str:
        """
        Main auto-reply path.
        Pipeline: intent → language → retrieve 15 → rerank to 5 → smart prompt → answer
        """
        if self.db is None:
            return ("Hi! I'm your AI assistant. No documents loaded yet. "
                    "Add your business docs to the docs/ folder and restart.")
        try:
            intent   = _cached_intent(self, question)
            language = self._detect_language(question)

            # Skip RAG for pure greetings — reply faster
            if intent == "greeting":
                lang_note = f"Reply in {language}." if language != "English" else ""
                msgs = [
                    {"role": "system",
                     "content": SYSTEM_PROMPT + "\nThis is a greeting — be warm, brief, mention 1-2 services. " + lang_note},
                    {"role": "user", "content": question},
                ]
                answer = self._ollama(msgs, temperature=0.4)
                return answer or "Hello! How can I help you today?"

            # RAG retrieval with reranking (#2)
            raw_docs = self.db.similarity_search(question, k=15)
            docs     = self._rerank_docs(question, raw_docs, top_k=5)
            context  = "\n\n".join(d.page_content for d in docs)

            messages = self._build_messages(
                question, context, intent, language,
                history, customer_profile
            )
            answer = self._ollama(messages, temperature=0.2)
            return answer or "Sorry, I ran into an issue. Please try again."

        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return "Sorry, I ran into an issue. Please try again."

    # ── Public: 3 suggestions ────────────────────────────────────────────────
    def suggestions(self, question: str, history: list = [],
                    customer_profile: dict = {}) -> list:
        """
        Generate 3 context-aware reply suggestions.
        Uses all enhancements: reranking, intent, language, profile, corrections.
        """
        if self.db is None:
            return [
                "Hi! How can I help you today?",
                "Sure, happy to assist!",
                "Thanks for reaching out. Let me help you.",
            ]
        try:
            intent   = _cached_intent(self, question)
            language = self._detect_language(question)

            raw_docs = self.db.similarity_search(question, k=15)
            docs     = self._rerank_docs(question, raw_docs, top_k=5)
            context  = "\n\n".join(d.page_content for d in docs)

            messages = self._build_messages(
                question, context, intent, language,
                history, customer_profile, for_suggestions=True
            )

            # Try with JSON format first
            raw = self._ollama(messages, temperature=0.7, json_format=True)
            result = _parse_suggestions(raw)
            if result:
                return result

            # Fallback without JSON format
            raw = self._ollama(messages, temperature=0.7)
            result = _parse_suggestions(raw)
            if result:
                return result

        except Exception as e:
            logger.error(f"Suggestions error: {e}")

        return [
            "Got it! I'll get back to you shortly.",
            "Sure, happy to help! What do you need?",
            "Thanks for reaching out. Let me look into that for you.",
        ]

    # ── Misc ──────────────────────────────────────────────────────────────────
    def ingest(self, text: str, source: str = "manual"):
        from langchain.schema import Document
        doc      = Document(page_content=text, metadata={"source": source})
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_documents([doc])
        if self.db:
            self.db.add_documents(chunks)
        else:
            self.db = FAISS.from_documents(chunks, self._embeddings())
        logger.info(f"Ingested {len(chunks)} chunks from '{source}'")

    def doc_count(self) -> int:
        try:
            return self.db.index.ntotal if self.db else 0
        except Exception:
            return -1

    def summarize_history(self, history: list) -> str:
        """Public endpoint for summarising a customer's old messages."""
        return self._summarize_old_messages(history)


# ── Helpers ───────────────────────────────────────────────────────────────────
_intent_cache: dict = {}
_engine_ref = None  # set by RAGEngine.__init__

def _cached_intent(engine, question: str) -> str:
    """Cache intent results to avoid repeated LLM calls for same question."""
    key = question[:100]
    if key not in _intent_cache:
        _intent_cache[key] = engine._detect_intent(question)
    return _intent_cache[key]


def _parse_suggestions(raw: str) -> list:
    """Parse a JSON array of 3 suggestions from LLM output."""
    if not raw:
        return []
    try:
        # Direct JSON parse
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) >= 2:
            result = [str(s).strip() for s in parsed[:3]]
            while len(result) < 3:
                result.append(result[0])
            return result
        if isinstance(parsed, dict):
            vals = [str(v).strip() for v in parsed.values() if v]
            if len(vals) >= 2:
                while len(vals) < 3:
                    vals.append(vals[0])
                return vals[:3]
    except json.JSONDecodeError:
        pass

    # Regex extraction
    match = re.search(r'\[[\s\S]*?\]', raw)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) >= 2:
                result = [str(s).strip() for s in arr[:3]]
                while len(result) < 3:
                    result.append(result[0])
                return result
        except Exception:
            pass

    # Line extraction
    lines = [re.sub(r'^[\d\*\-•]+[\.\):\s]+', '', l).strip() for l in raw.split('\n')]
    lines = [l for l in lines if 5 < len(l) < 300]
    if len(lines) >= 2:
        while len(lines) < 3:
            lines.append(lines[0])
        return lines[:3]

    return []
