# Web3 AI RAG Template

A fully local, free-to-run Retrieval-Augmented Generation (RAG) template
for querying Web3 whitepaper PDFs with an LLM.

## 🎯 Features
- **PDF ingestion:** Automatically loads every PDF in `docs/` using pypdf.
- **Local embeddings:** Uses `BAAI/bge-small-en-v1.5` via sentence-transformers.
- **Local vector store:** ChromaDB with persistent storage.
- **Local LLM:** Ollama running Llama 3.2 — no API keys, no cost.
- **Source citations:** Every answer references the originating PDF.

## 🛠️ Tech Stack
| Component | Tool |
|-----------|------|
| Embedding | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB |
| LLM | Ollama + Llama 3.2 |
| PDF parsing | pypdf |
| Language | Python 3.10+ |

## 📂 Project Structure