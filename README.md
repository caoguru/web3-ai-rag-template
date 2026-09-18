# web3-ai-rag-template

A 100% free, fully local RAG (Retrieval-Augmented Generation) template designed to analyze and answer questions based on Web3 project whitepapers.

## Tech Stack
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Model:** BAAI/bge-small-en-v1.5 (Open-source via HuggingFace)
- **Local LLM:** Ollama (Llama 3.2)

## Prerequisites
1. Install [Ollama](https://ollama.com).
2. Download the Llama 3.2 model locally:
   ```bash
   ollama run llama3.2