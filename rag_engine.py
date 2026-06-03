"""
RAG Engine — loads documents, embeds them, and answers questions.
Drop any .txt, .pdf, or .md files into the `docs/` folder.
Uses Ollama for both embeddings and LLM (no OpenAI required).
"""
import os
import glob
import logging
import httpx
from typing import Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

DOCS_DIR          = os.getenv("DOCS_DIR", "docs")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant. "
    "Answer the question using only the context below. "
    "If the answer is not in the context, say: "
    "'I don't have that information right now. Please contact us directly.' "
    "Keep your reply short and conversational (max 3 sentences). Do not use markdown."
)


class RAGEngine:
    def __init__(self):
        self.db: Optional[FAISS] = None
        self._load()

    def _embeddings(self):
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def _load(self):
        os.makedirs(DOCS_DIR, exist_ok=True)
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                self.db = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self._embeddings(),
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded vector store from {VECTOR_STORE_PATH}")
                return
            except Exception as e:
                logger.warning(f"Could not load store: {e}. Rebuilding...")
        self._build_from_docs()

    def _build_from_docs(self):
        docs = []
        for pattern in [f"{DOCS_DIR}/**/*.txt", f"{DOCS_DIR}/**/*.md"]:
            for f in glob.glob(pattern, recursive=True):
                try:
                    docs.extend(TextLoader(f, encoding="utf-8").load())
                except Exception as e:
                    logger.warning(f"Skipping {f}: {e}")

        for f in glob.glob(f"{DOCS_DIR}/**/*.pdf", recursive=True):
            try:
                docs.extend(PyPDFLoader(f).load())
            except Exception as e:
                logger.warning(f"Skipping {f}: {e}")

        if not docs:
            logger.warning("No documents in docs/. Using fallback mode.")
            self.db = None
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        logger.info(f"Indexed {len(chunks)} chunks from {len(docs)} documents")

        self.db = FAISS.from_documents(chunks, self._embeddings())
        self.db.save_local(VECTOR_STORE_PATH)

    def _ollama_chat(self, context: str, question: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except httpx.ConnectError:
            return "Ollama is not running. Please start it with: ollama serve"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Sorry, I ran into an issue. Please try again."

    def query(self, question: str) -> str:
        if self.db is None:
            return (
                "Hi! I'm your AI assistant. No documents loaded yet. "
                "Add your business docs to the docs/ folder and restart."
            )
        try:
            docs = self.db.similarity_search(question, k=4)
            context = "\n\n".join(d.page_content for d in docs)
            return self._ollama_chat(context, question)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "Sorry, I ran into an issue. Please try again."

    def ingest(self, text: str, source: str = "manual"):
        from langchain.schema import Document
        doc = Document(page_content=text, metadata={"source": source})
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc])
        if self.db:
            self.db.add_documents(chunks)
        else:
            self.db = FAISS.from_documents(chunks, self._embeddings())
        logger.info(f"Ingested {len(chunks)} new chunks from '{source}'")

    def doc_count(self) -> int:
        if self.db:
            try:
                return self.db.index.ntotal
            except Exception:
                return -1
        return 0
