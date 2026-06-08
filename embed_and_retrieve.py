"""
Embedding & Retrieval Pipeline
================================
Loads chunks from the ingestion pipeline, embeds with SPECTER2,
stores in ChromaDB with source metadata, and provides retrieval.

SPECTER2 uses the `adapters` library (not sentence-transformers).
The proximity adapter is best for retrieval tasks on scientific text.

Usage:
    # Build the vector store (run once, or after re-chunking)
    python embed_and_retrieve.py --build

    # Query interactively
    python embed_and_retrieve.py --query "What is the BAM model?"

    # Run evaluation queries
    python embed_and_retrieve.py --eval
"""

import json
import argparse
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import chromadb


# ── Configuration ──────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
CHUNKS_FILE = PROJECT_DIR / "chunks" / "chunks.json"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "human_ai_adaptation"
BASE_MODEL = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"       # proximity adapter for retrieval
TOP_K = 5
BATCH_SIZE = 16
MAX_LENGTH = 512  # SPECTER2's max token length

# Evaluation queries from planning.md
EVAL_QUERIES = [
    {
        "id": 1,
        "query": "What is the Bounded-Memory Adaptation Model (BAM) and how does it enable mutual adaptation?",
        "expected_sources": [
            "nikolaidis2016bam.pdf",
            "nikolaidis-et-al-2017-human-robot-mutual-adaptation-in-collaborative-tasks-models-and-experiments.pdf",
        ],
        "expected_keywords": ["bounded memory", "POMDP", "adaptab", "trust"],
    },
    {
        "id": 2,
        "query": "What are the three classes of latent dynamics models used in human-robot mutual adaptation?",
        "expected_sources": [
            "Mutual Adaptation and Influence Survey of Latent Dynamics Models in HRI.pdf",
        ],
        "expected_keywords": ["Bayesian", "Markov", "encoded", "prediction", "control"],
    },
    {
        "id": 3,
        "query": "How does correlation neglect affect human-AI collaboration in the Bayesian framework?",
        "expected_sources": ["2602.14331v1.pdf"],
        "expected_keywords": ["correlation", "overlapping", "complementarity"],
    },
    {
        "id": 4,
        "query": "What are the five requirements for successful human-robot co-learning?",
        "expected_sources": [
            "Mapping Human-Agent Co-Learning and Co-Adaptation A Scoping Review.pdf",
        ],
        "expected_keywords": ["shared goal", "synchrony", "interdependence", "adaptab", "transparency"],
    },
    {
        "id": 5,
        "query": (
            "How does the system-theoretical approach characterize human interaction "
            "with agentic AI differently from traditional HCI frameworks?"
        ),
        "expected_sources": ["fhumd-7-1579166.pdf"],
        "expected_keywords": ["dynamic system", "feedback", "emergent", "complex adaptive"],
    },
]


# ── 1. Load Chunks ───────────────────────────────────────────────

def load_chunks(chunks_file: Path = CHUNKS_FILE) -> list[dict]:
    """Load chunks from the JSON file produced by ingest_and_chunk.py."""
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_file}\n"
            "Run `python ingest_and_chunk.py` first to generate chunks."
        )
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks from {chunks_file.name}")
    return chunks


# ── 2. Embedding Model ──────────────────────────────────────────

def load_model(
    base_model: str = BASE_MODEL,
    adapter_name: str = ADAPTER_NAME,
):
    """
    Load SPECTER2 base model + proximity adapter.
    Returns (tokenizer, model).
    """
    print(f"  Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"  Loading base model: {base_model}")
    model = AutoAdapterModel.from_pretrained(base_model)

    print(f"  Loading adapter: {adapter_name}")
    loaded_name = model.load_adapter(
        adapter_name, source="hf", load_as="specter2", set_active=True,
    )
    model.set_active_adapters(loaded_name)
    print(f"  Active adapters: {model.active_adapters}")

    model.eval()
    print(f"  Model ready (max_length={MAX_LENGTH})")
    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Embed a list of texts using SPECTER2.
    Returns numpy array of shape (n_texts, hidden_dim).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        with torch.no_grad():
            output = model(**inputs)

        # CLS token embedding (first token)
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"    Embedded {i}/{len(texts)} chunks...")

    return np.concatenate(all_embeddings, axis=0)


# ── 3. Build Vector Store ────────────────────────────────────────

def build_vector_store(
    chunks: list[dict],
    tokenizer,
    model,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Embed all chunks and store in a persistent ChromaDB collection.
    Metadata per chunk: source_file, source_title, chunk_index.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts, tokenizer, model)
    print(f"  Embedding shape: {embeddings.shape}")

    # Set up persistent ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'")
    except (ValueError, Exception) as e:
        # ChromaDB raises NotFoundError (subclass of Exception) on missing collection
        if "does not exist" in str(e) or "NotFound" in type(e).__name__:
            pass
        else:
            raise

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches (ChromaDB batch limit)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end]
        batch_embeddings = embeddings[i:end].tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {
                    "source_file": c["source_file"],
                    "source_title": c["source_title"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "char_count": c["char_count"],
                }
                for c in batch_chunks
            ],
        )

    print(f"  Stored {collection.count()} chunks in ChromaDB at {chroma_dir}")
    return collection


# ── 4. Retrieval ─────────────────────────────────────────────────

def retrieve(
    query: str,
    tokenizer,
    model,
    collection: chromadb.Collection,
    k: int = TOP_K,
) -> list[dict]:
    """
    Embed the query with SPECTER2, retrieve top-k similar chunks
    from ChromaDB with distance scores and metadata.
    """
    query_embedding = embed_texts([query], tokenizer, model).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return retrieved


def print_results(query: str, results: list[dict], verbose: bool = True):
    """Pretty-print retrieval results."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n--- Result {i}/{len(results)} "
              f"(distance: {r['distance']:.4f}) ---")
        print(f"  Source: {meta['source_file']}")
        print(f"  Chunk:  {meta['chunk_index']}/{meta['total_chunks'] - 1}")
        print(f"  Chars:  {meta['char_count']}")
        if verbose:
            text = r["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"  Text:   {text}")
    print()


# ── 5. Evaluation ───────────────────────────────────────────────

def run_eval(
    tokenizer,
    model,
    collection: chromadb.Collection,
    queries: list[dict] = EVAL_QUERIES,
    k: int = TOP_K,
):
    """
    Run evaluation queries and check:
    - Are distances below 0.5?
    - Do results come from expected source papers?
    - Do chunks contain expected keywords?
    """
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    all_pass = True

    for eq in queries:
        results = retrieve(eq["query"], tokenizer, model, collection, k)
        print_results(eq["query"], results, verbose=True)

        # Check 1: Distance scores
        distances = [r["distance"] for r in results]
        best_dist = distances[0]
        if best_dist > 0.5:
            print(f"  WARNING: Best distance {best_dist:.4f} > 0.5 -- weak match")
            all_pass = False
        else:
            print(f"  PASS: Best distance {best_dist:.4f} < 0.5")

        # Check 2: Source papers
        result_sources = {r["metadata"]["source_file"] for r in results}
        expected_hit = any(
            src in result_sources for src in eq["expected_sources"]
        )
        if expected_hit:
            matched = result_sources & set(eq["expected_sources"])
            print(f"  PASS: Expected source(s) found: {matched}")
        else:
            print(f"  WARNING: Expected sources {eq['expected_sources']} "
                  f"not in results: {result_sources}")
            all_pass = False

        # Check 3: Keyword presence in top results
        top_text = " ".join(r["text"].lower() for r in results[:3])
        found_kw = [kw for kw in eq["expected_keywords"]
                     if kw.lower() in top_text]
        missing_kw = [kw for kw in eq["expected_keywords"]
                       if kw.lower() not in top_text]
        if found_kw:
            print(f"  Keywords found: {found_kw}")
        if missing_kw:
            print(f"  Keywords missing: {missing_kw}")

        print()

    if all_pass:
        print("ALL CHECKS PASSED -- retrieval is working correctly.")
    else:
        print("SOME CHECKS FAILED -- review warnings above before proceeding.")

    return all_pass


# ── 6. Connect to Existing Store ─────────────────────────────────

def get_collection(
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Connect to an existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(collection_name)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embedding & Retrieval Pipeline"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build the vector store from chunks (run after ingest_and_chunk.py)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single retrieval query",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run all evaluation queries from planning.md",
    )
    parser.add_argument(
        "--k", type=int, default=TOP_K,
        help=f"Number of results to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    if not any([args.build, args.query, args.eval]):
        parser.print_help()
        return

    # Load the embedding model (needed for all operations)
    print("\n[1] Loading embedding model...")
    tokenizer, model = load_model()

    if args.build:
        # Full build: load chunks -> embed -> store
        print("\n[2] Loading chunks...")
        chunks = load_chunks()
        print("\n[3] Building vector store...")
        collection = build_vector_store(chunks, tokenizer, model)
        print("\nVector store built successfully.")

        if not args.query and not args.eval:
            # Quick sanity check after build
            print("\n[4] Sanity check -- running one test query...")
            test_q = "What is the Bounded-Memory Adaptation Model?"
            results = retrieve(test_q, tokenizer, model, collection, k=3)
            print_results(test_q, results, verbose=False)
    else:
        # Connect to existing store
        print("\n[2] Connecting to existing vector store...")
        collection = get_collection()
        print(f"  Collection '{COLLECTION_NAME}': {collection.count()} chunks")

    if args.query:
        results = retrieve(args.query, tokenizer, model, collection, k=args.k)
        print_results(args.query, results)

    if args.eval:
        run_eval(tokenizer, model, collection, k=args.k)


if __name__ == "__main__":
    main()
