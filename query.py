"""
Generation Pipeline
====================
Wires retrieval to LLM generation with enforced grounding and
programmatic source attribution.

Usage:
    # Interactive CLI
    python query.py "What is the BAM model?"

    # Run grounding tests
    python query.py --test
"""

import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from embed_and_retrieve import (
    load_model,
    get_collection,
    retrieve,
    TOP_K,
)

# ── Configuration ──────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K_RETRIEVE = TOP_K  # 5, from embed_and_retrieve


# ── System Prompt ────────────────────────────────────────────────
# This prompt ENFORCES grounding — the model is forbidden from
# using its own knowledge. It must answer ONLY from the provided
# documents and cite them by their [SOURCE N] labels.

SYSTEM_PROMPT = """\
You are a research assistant for a literature review on human-AI mutual \
adaptation and complex adaptive systems.

STRICT RULES — you must follow ALL of these:

1. ONLY use information from the DOCUMENTS provided below. Do NOT use \
your own training knowledge, background assumptions, or general \
information about the topic.

2. If the documents do not contain enough information to answer the \
question, respond EXACTLY with: "I don't have enough information in \
the provided documents to answer that question."

3. When answering, reference the source documents by their [SOURCE N] \
labels (e.g., "According to [SOURCE 1], ..."). Use multiple sources \
when the answer draws from more than one document.

4. Keep answers concise and directly responsive to the question. Do \
not add information beyond what the documents state.

5. If only partial information is available, answer what you can and \
explicitly note what the documents do not cover.
"""


def _build_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Each chunk is labeled [SOURCE N] with its filename so the LLM
    can reference them and we can verify attribution.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"]["source_file"]
        context_parts.append(
            f"[SOURCE {i}] (from: {source})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def _extract_cited_sources(
    answer: str,
    retrieved_chunks: list[dict],
) -> list[str]:
    """
    Programmatically extract which sources the LLM actually cited.
    Returns deduplicated list of source filenames.
    Also always includes ALL retrieved source files as 'retrieved from'
    context — source attribution is not left to the LLM alone.
    """
    import re

    cited = set()
    # Find [SOURCE N] references in the answer
    refs = re.findall(r"\[SOURCE\s*(\d+)\]", answer)
    for ref in refs:
        idx = int(ref) - 1
        if 0 <= idx < len(retrieved_chunks):
            cited.add(retrieved_chunks[idx]["metadata"]["source_file"])

    return sorted(cited)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    client: Groq | None = None,
    model: str = GROQ_MODEL,
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Returns:
        {
            "answer": str,           # The LLM's response
            "sources_cited": [str],  # Files the LLM cited via [SOURCE N]
            "sources_retrieved": [str],  # All files chunks came from
            "chunks_used": int,      # Number of chunks in context
        }
    """
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "Set GROQ_API_KEY in your .env file. "
                "Get a free key at https://console.groq.com"
            )
        client = Groq(api_key=api_key)

    context = _build_context(retrieved_chunks)

    user_message = f"DOCUMENTS:\n\n{context}\n\nQUESTION: {query}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Low temperature for factual grounding
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # Programmatic source attribution
    sources_cited = _extract_cited_sources(answer, retrieved_chunks)
    sources_retrieved = sorted(
        {c["metadata"]["source_file"] for c in retrieved_chunks}
    )

    return {
        "answer": answer,
        "sources_cited": sources_cited,
        "sources_retrieved": sources_retrieved,
        "chunks_used": len(retrieved_chunks),
    }


# ── End-to-End Ask Function ──────────────────────────────────────

# Cache the model and collection so repeated calls don't reload
_model_cache = {}


def _get_resources():
    """Lazy-load and cache the embedding model and ChromaDB collection."""
    if "tokenizer" not in _model_cache:
        print("Loading embedding model...")
        tokenizer, model = load_model()
        collection = get_collection()
        _model_cache["tokenizer"] = tokenizer
        _model_cache["model"] = model
        _model_cache["collection"] = collection
        print(f"Ready — {collection.count()} chunks in vector store")
    return (
        _model_cache["tokenizer"],
        _model_cache["model"],
        _model_cache["collection"],
    )


def ask(question: str, k: int = TOP_K_RETRIEVE) -> dict:
    """
    End-to-end: question -> retrieve -> generate -> return.

    Returns:
        {
            "answer": str,
            "sources_cited": [str],
            "sources_retrieved": [str],
            "chunks_used": int,
            "distances": [float],
        }
    """
    tokenizer, emb_model, collection = _get_resources()

    # Retrieve
    retrieved = retrieve(question, tokenizer, emb_model, collection, k=k)
    distances = [r["distance"] for r in retrieved]

    # Generate
    result = generate_answer(question, retrieved)
    result["distances"] = distances

    return result


# ── CLI & Tests ──────────────────────────────────────────────────

def run_grounding_tests():
    """
    Test grounding with in-scope and out-of-scope queries.
    Verifies the system uses only retrieved context and declines
    when documents don't cover a topic.
    """
    print("\n" + "=" * 70)
    print("GROUNDING TESTS")
    print("=" * 70)

    tests = [
        {
            "query": "What is the Bounded-Memory Adaptation Model (BAM) and how does it enable mutual adaptation?",
            "should_answer": True,
            "check": "Should reference BAM, POMDP, adaptability, trust",
        },
        {
            "query": "How does correlation neglect affect human-AI collaboration in the Bayesian framework?",
            "should_answer": True,
            "check": "Should reference correlation neglect, overlapping signals",
        },
        {
            "query": "What is the capital of France and what are popular tourist attractions there?",
            "should_answer": False,
            "check": "Should decline — documents don't cover French geography",
        },
    ]

    for i, test in enumerate(tests, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}: {test['query']}")
        print(f"Expected: {'ANSWER' if test['should_answer'] else 'DECLINE'}")
        print(f"Check: {test['check']}")
        print(f"{'─'*70}")

        result = ask(test["query"])

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources cited by LLM: {result['sources_cited']}")
        print(f"Sources retrieved:     {result['sources_retrieved']}")
        print(f"Distances:             {[f'{d:.4f}' for d in result['distances']]}")

        # Check grounding
        declined = "don't have enough information" in result["answer"].lower()
        if test["should_answer"] and declined:
            print("WARNING: System declined but should have answered")
        elif not test["should_answer"] and not declined:
            print("WARNING: System answered but should have declined")
        else:
            print("PASS")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--test", action="store_true",
        help="Run grounding tests (in-scope + out-of-scope)",
    )
    parser.add_argument(
        "--k", type=int, default=TOP_K_RETRIEVE,
        help=f"Number of chunks to retrieve (default: {TOP_K_RETRIEVE})",
    )
    args = parser.parse_args()

    if args.test:
        run_grounding_tests()
    elif args.question:
        result = ask(args.question, k=args.k)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources cited: {result['sources_cited']}")
        print(f"Sources retrieved: {result['sources_retrieved']}")
        print(f"Distances: {[f'{d:.4f}' for d in result['distances']]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
