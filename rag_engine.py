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
import hashlib
import threading
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
OLLAMA_FAST_MODEL  = os.getenv("OLLAMA_FAST_MODEL",  OLLAMA_MODEL)  # smaller model for suggestions
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

INTENT_TYPES = {"greeting", "pricing", "complaint", "followup", "closing", "general"}

SYSTEM_PROMPT = (
    "You are a WhatsApp assistant for a business. "
    "STRICT RULES — follow these no matter what the user asks:\n"
    "1. Reply in 2-3 SHORT sentences only. Never more. No exceptions.\n"
    "2. No bullet points, no numbered lists, no markdown, no long explanations.\n"
    "3. If the user asks to 'explain in detail' or 'elaborate', still reply in 2-3 sentences "
    "and say they can ask follow-up questions for more.\n"
    "4. Answer using ONLY facts explicitly written in the Business Context below. "
    "NEVER guess, invent, or add any detail that is not clearly stated in the context. "
    "If the exact answer is not in the context, say exactly: "
    "'I don't have that information right now. Please contact us directly.'"
)

SUGGESTIONS_SYSTEM_PROMPT = (
    "You are a WhatsApp business assistant. Write 3 short reply options for the agent to send.\n"
    "Rules:\n"
    "1. Use ONLY facts from the Business Context — do not invent anything.\n"
    "2. Each reply: 1 sentence, plain text, no markdown, no 'A:' prefix.\n"
    "3. Give 3 different phrasings of the same answer.\n"
    "4. The Business Context always has the answer — never say 'I don't have that information'.\n"
    "5. Output exactly like this example (replace the example text with your actual replies):\n"
    "1. Our platform is completely free to use.\n"
    "2. No payment required, all features are free.\n"
    "3. You can use everything on the platform at no cost."
)


# ── RAG Engine ────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        self.db: Optional[FAISS] = None
        self._corrections: list  = self._load_corrections()
        self._summary_cache: dict          = {}
        self._summary_lock: threading.Lock = threading.Lock()
        # Only one summary runs at a time — Ollama is single-threaded
        self._summary_sem: threading.Semaphore = threading.Semaphore(1)
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
    def _detect_intent(self, question: str, use_llm: bool = True) -> str:
        """
        Classify the message intent so the reply prompt can be tailored.
        Uses fast keyword heuristics first; falls back to a tiny Ollama call
        only for ambiguous messages when use_llm=True.
        Pass use_llm=False (suggestions path) to skip the LLM call entirely.
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

        if not use_llm:
            return "general"

        # LLM fallback for ambiguous messages (auto-reply path only)
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
            with httpx.Client(timeout=120.0) as client:
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

    def _summary_cache_key(self, old_msgs: list) -> str:
        if not old_msgs:
            return ""
        text = "\n".join(f"{m['role']}:{m['content']}" for m in old_msgs)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _run_summary_background(self, old_msgs: list, cache_key: str) -> None:
        # Skip immediately if another summary is already running
        acquired = self._summary_sem.acquire(blocking=False)
        if not acquired:
            logger.debug("Summary skipped — another summary already running")
            return
        try:
            # Brief pause so the main Ollama chat call can finish first
            import time
            time.sleep(5)
            summary = self._summarize_old_messages(old_msgs)
            if summary:
                with self._summary_lock:
                    self._summary_cache[cache_key] = summary
                logger.info(f"Background summary cached (key={cache_key[:8]}…, {len(old_msgs)} msgs)")
        finally:
            self._summary_sem.release()

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

        if for_suggestions:
            system = (
                SUGGESTIONS_SYSTEM_PROMPT
                + (f"\n{lang_note}" if lang_note else "")
                + profile_block
                + correction_block
            )
        else:
            system = (
                SYSTEM_PROMPT
                + (f"\n{tone}" if tone else "")
                + (f"\n{lang_note}" if lang_note else "")
                + profile_block
                + correction_block
            )

        # History with summary (#9)
        # Split: summarise old messages, keep last 10 verbatim
        prior = [m for m in history if not (m["role"] == "user" and m["content"] == question)]
        old_msgs    = prior[:-10] if len(prior) > 10 else []
        recent_msgs = prior[-10:]

        messages = [{"role": "system", "content": system}]

        # Inject summary if we have old messages (never block — read from cache only)
        summary = customer_profile.get("summary", "") if customer_profile else ""
        if not summary and old_msgs:
            cache_key = self._summary_cache_key(old_msgs)
            with self._summary_lock:
                summary = self._summary_cache.get(cache_key, "")

        if summary:
            messages.append({
                "role":    "system",
                "content": f"[Earlier conversation summary: {summary}]"
            })

        messages.extend(recent_msgs)
        user_content = f"Business Context:\n{context}\n\nCustomer message: {question}"
        if for_suggestions:
            user_content += '\n\nWrite the 3 reply options now:'
        messages.append({"role": "user", "content": user_content})

        return messages

    # ── Ollama call ───────────────────────────────────────────────────────────
    def _ollama(self, messages: list, temperature: float = 0.2,
                json_format: bool = False, num_predict: int = None,
                model: str = None) -> str:
        payload = {
            "model":    model or OLLAMA_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature},
        }
        if num_predict:
            payload["options"]["num_predict"] = num_predict
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

            # Skip RAG for pure greetings / closings — reply faster
            if intent == "greeting":
                lang_note = f"Reply in {language}." if language != "English" else ""
                msgs = [
                    {"role": "system",
                     "content": SYSTEM_PROMPT + "\nThis is a greeting — be warm, brief, mention 1-2 services. " + lang_note},
                    {"role": "user", "content": question},
                ]
                answer = self._ollama(msgs, temperature=0.4, num_predict=120)
                return answer or "Hello! How can I help you today?"

            if intent == "closing":
                lang_note = f"Reply in {language}." if language != "English" else ""
                msgs = [
                    {"role": "system",
                     "content": SYSTEM_PROMPT + "\nConversation is wrapping up — give a warm, very brief sign-off and invite future contact. " + lang_note},
                    {"role": "user", "content": question},
                ]
                answer = self._ollama(msgs, temperature=0.4, num_predict=80)
                return answer or "Thank you! Feel free to reach out anytime."

            # RAG retrieval with reranking (#2)
            raw_docs = self.db.similarity_search(question, k=15)
            docs     = self._rerank_docs(question, raw_docs, top_k=5)
            context  = "\n\n".join(d.page_content for d in docs)

            messages = self._build_messages(
                question, context, intent, language,
                history, customer_profile
            )
            answer = self._ollama(messages, temperature=0.2, num_predict=150)
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
            # Skip LLM intent call for suggestions — heuristics only (saves 8s)
            intent   = self._detect_intent(question, use_llm=False)
            language = self._detect_language(question)

            # Greeting / closing: skip RAG entirely — return intent-appropriate options instantly
            if intent == "greeting":
                return [
                    "Hello! Welcome to Portfolio Simulator. How can I help you?",
                    "Hi there! How can I assist you with your investment queries today?",
                    "Hey! Great to hear from you. What would you like to know?",
                ]
            if intent == "closing":
                return [
                    "You're welcome! Feel free to reach out anytime.",
                    "Happy to help! Let us know if you have more questions.",
                    "Anytime! Have a great day.",
                ]

            raw_docs = self.db.similarity_search(question, k=20)
            docs     = self._rerank_docs(question, raw_docs, top_k=6)
            context  = "\n\n".join(d.page_content for d in docs)

            messages = self._build_messages(
                question, context, intent, language,
                history, customer_profile, for_suggestions=True
            )

            # Use fast model for suggestions (no json_format — small models output JSON Schema objects)
            raw = self._ollama(messages, temperature=0.4, num_predict=200,
                               model=OLLAMA_FAST_MODEL)
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


def _clean(s: str) -> str:
    """Strip surrounding quotes, FAQ prefixes, and JSON artifact suffixes."""
    s = s.strip().strip('"').strip("'")
    s = re.sub(r'^[AaQq]\s*:\s*', '', s)           # strip "A: " / "Q: " FAQ prefix
    s = re.sub(r'["\s]*:\s*[\[\]{},]*\s*$', '', s) # strip "":[] suffix
    return s.strip()


def _is_valid_suggestion(s: str) -> bool:
    """Reject URLs, JSON schema metadata, and non-reply strings."""
    if not s or len(s) < 8 or len(s) > 300:
        return False
    if s.startswith(('http://', 'https://', '$', '#')):
        return False
    if s.lower() in ('array', 'string', 'object', 'number', 'boolean', 'null', 'integer',
                     'true', 'false', 'items', 'properties', 'type'):
        return False
    return True


def _parse_suggestions(raw: str) -> list:
    """Parse 3 suggestion strings from LLM output — handles array, dict, or plain lines."""
    if not raw:
        return []

    # Helper: filter and pad a candidate list
    def _finalize(candidates: list) -> list:
        valid = [s for s in candidates if _is_valid_suggestion(s)]
        if len(valid) < 2:
            return []
        while len(valid) < 3:
            valid.append(valid[0])
        return valid[:3]

    try:
        parsed = json.loads(raw)
        # ["r1","r2","r3"]
        if isinstance(parsed, list) and len(parsed) >= 2:
            result = _finalize([_clean(str(s)) for s in parsed[:5] if str(s).strip()])
            if result:
                return result
        if isinstance(parsed, dict):
            # {"r1":"r2","r3":...} — values are the replies
            vals = [_clean(str(v)) for v in parsed.values()
                    if str(v).strip() and str(v).strip() not in ('[]', '{}', 'null', 'None')]
            result = _finalize(vals)
            if result:
                return result
            # llama3.2 sometimes puts replies as KEYS with empty values — use keys
            keys = [_clean(str(k)) for k in parsed.keys() if len(str(k)) > 8]
            result = _finalize(keys)
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Regex: extract first JSON array in output
    match = re.search(r'\[[\s\S]*?\]', raw)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) >= 2:
                result = _finalize([_clean(str(s)) for s in arr[:5] if str(s).strip()])
                if result:
                    return result
        except Exception:
            pass

    # Line extraction — strip numbering AND trailing JSON artifacts
    lines = []
    for l in raw.split('\n'):
        l = re.sub(r'^[\d\*\-•]+[\.\):\s]+', '', l)  # strip "1. " prefix
        l = re.sub(r'["\s]*:\s*[\[\]{},]*\s*$', '', l)  # strip "":[] suffix
        l = l.strip().strip('"').strip("'")
        if _is_valid_suggestion(l):
            lines.append(l)
    return _finalize(lines)
