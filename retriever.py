"""
retriever.py
Query the ChromaDB collection and return the most relevant chunks
for a given user question.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "web3_whitepapers"
TOP_K = 5


class Retriever:
    """Encapsulates embedding model and ChromaDB collection access."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: User question in English.
            top_k: Number of chunks to retrieve.

        Returns:
            List of dicts with 'text', 'source', and 'distance'.
        """
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": text,
                "source": meta["source"],
                "distance": dist,
            })
        return hits


if __name__ == "__main__":
    retriever = Retriever()
    query = "What is the consensus mechanism of Ethereum?"
    hits = retriever.search(query)

    print(f"Query: {query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] source={hit['source']} distance={hit['distance']:.4f}")
        print(f"    {hit['text'][:200]}...\n")