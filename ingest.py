"""
ingest.py — CLI tool to rebuild the vector store from docs/
Run: python ingest.py
"""
from rag_engine import RAGEngine
import logging
logging.basicConfig(level=logging.INFO)

print("🔄 Rebuilding vector store from docs/...")
import shutil, os
if os.path.exists("vector_store"):
    shutil.rmtree("vector_store")

engine = RAGEngine()
print(f"✅ Done! {engine.doc_count()} vectors indexed.")
