"""
chatbot.py
RAG chatbot: retrieve relevant chunks from ChromaDB,
then generate an answer with Ollama (Llama 3.2).
"""

import ollama
from retriever import Retriever

LLM_MODEL = "llama3.2"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful Web3 assistant. Answer the user's question "
    "using ONLY the provided context. If the answer is not in the "
    "context, say you don't know. Always cite the source document name."
)


def build_context(hits: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string.

    Args:
        hits: List of retrieved chunks with 'text' and 'source'.

    Returns:
        A formatted context string.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(f"[Source {i}: {hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def answer(question: str, retriever: Retriever) -> str:
    """
    Answer a user question using RAG.

    Args:
        question: User question in English.
        retriever: Initialized Retriever instance.

    Returns:
        The model's answer as a string.
    """
    hits = retriever.search(question, top_k=TOP_K)
    context = build_context(hits)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def main() -> None:
    """Run an interactive RAG chat loop in the terminal."""
    print("Web3 AI RAG Chatbot (type 'exit' to quit)\n")
    retriever = Retriever()

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not question:
            continue

        print("\nThinking...\n")
        reply = answer(question, retriever)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()