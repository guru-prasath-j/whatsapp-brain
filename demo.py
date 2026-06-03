"""
demo.py — Test the RAG engine locally without WhatsApp
Run: python demo.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from rag_engine import RAGEngine
import logging
logging.basicConfig(level=logging.WARNING)

print("🤖 WhatsApp Brain — Local Demo")
print("=" * 40)
print("Type a question to test the RAG engine.")
print("Type 'quit' to exit.\n")

engine = RAGEngine()
print(f"✅ Vector store loaded — {engine.doc_count()} vectors\n")

while True:
    try:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer = engine.query(q)
        print(f"Bot: {answer}\n")
    except KeyboardInterrupt:
        break

print("Goodbye!")
