"""
embedder.py
Chunk all PDFs from docs/, generate embeddings with
BAAI/bge-small-en-v1.5, and store them in a persistent ChromaDB collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from loader import load_all_documents

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "web3_whitepapers"

MAX_CHARS = 800
OVERLAP = 100


def smart_chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by paragraph, respecting max size.

    Args:
        text: Full document text.
        max_chars: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i:i + max_chars])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def build_vector_store() -> None:
    """
    Load all PDFs, chunk them, embed them, and store in ChromaDB.
    Existing collection is deleted first for a clean rebuild.
    """
    print("Loading documents...")
    documents = load_all_documents()

    print(f"\nLoading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for doc in documents:
        source = doc["source"]
        chunks = smart_chunk(doc["text"])
        print(f"[CHUNK] {source}: {len(chunks)} chunks")

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\nTotal chunks stored: {total_chunks}")
    print(f"Vector store path: {CHROMA_PATH}")


if __name__ == "__main__":
    build_vector_store()