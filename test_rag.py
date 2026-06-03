"""
test_rag.py — Quick local test of the RAG engine without running the server.
Run: python test_rag.py
"""
import httpx
import logging
logging.basicConfig(level=logging.WARNING)

# ── Step 1: Check Ollama and list available models ────────────────────────────
print("=" * 55)
print("   WhatsApp Brain — RAG Engine Test")
print("=" * 55)

try:
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"\nOllama is running. Available models: {models}\n")
except Exception as e:
    print(f"\n[!] Cannot reach Ollama: {e}")
    print("    Make sure Ollama is running: ollama serve")
    exit(1)

if not models:
    print("[!] No models found. Pull one first, e.g.: ollama pull llama3")
    exit(1)

# Auto-pick model: prefer llama3 variants, else use first available
import os
configured = os.getenv("OLLAMA_MODEL", "llama3")
if configured in models:
    chosen = configured
else:
    # pick best available
    preferred = ["llama3", "llama3.2", "llama3:latest", "llama3.2:latest",
                 "llama3:8b", "llama3.1", "mistral", "phi3", "gemma2"]
    chosen = next((m for m in preferred if m in models), models[0])
    print(f"[!] '{configured}' not found. Using '{chosen}' instead.")
    print(f"    Tip: update OLLAMA_MODEL in .env to '{chosen}'\n")

os.environ["OLLAMA_MODEL"] = chosen

# ── Step 2: Build vector store ────────────────────────────────────────────────
import shutil
if os.path.exists("vector_store"):
    shutil.rmtree("vector_store")

from rag_engine import RAGEngine
print("Loading documents from docs/ and building vector store...")
engine = RAGEngine()
count = engine.doc_count()
print(f"✅ {count} vectors indexed.\n")

if count == 0:
    print("⚠️  No vectors — make sure docs/ has your company files.")
    exit(1)

# ── Step 3: Run test queries ──────────────────────────────────────────────────
test_questions = [
    "What services do you offer?",
    "How much does a mobile app cost?",
    "What are your working hours?",
    "Do you offer free consultations?",
    "How long does it take to build a website?",
    "What payment methods do you accept?",
    "Do you work with international clients?",
]

print(f"Running test queries using model: {chosen}\n")
for q in test_questions:
    print(f"Q: {q}")
    answer = engine.query(q)
    print(f"A: {answer}")
    print("-" * 50)
