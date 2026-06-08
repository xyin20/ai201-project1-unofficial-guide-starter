"""
Ingestion and Chunking Pipeline
================================
Loads PDFs from documents/, cleans text, and produces chunks
matching the spec: ~512 tokens (~2000 chars), 128-token overlap (~500 chars).
Splits on paragraph boundaries first, falls back to character-count splits.

Usage:
    python ingest_and_chunk.py
    python ingest_and_chunk.py --chunk-size 2000 --overlap 500
"""

import os
import re
import json
import html
import argparse
from pathlib import Path
import pdfplumber


# ── Configuration ──────────────────────────────────────────────────

DOCUMENTS_DIR = Path(__file__).parent / "documents"
RAW_TEXT_DIR = Path(__file__).parent / "raw_text"
OUTPUT_DIR = Path(__file__).parent / "chunks"
CHUNK_SIZE = 2000       # ~512 tokens in characters
CHUNK_OVERLAP = 500     # ~128 tokens in characters


# ── 1. Document Ingestion ─────────────────────────────────────────

def ingest_documents(docs_dir: Path) -> list[dict]:
    """
    Load all PDFs from docs_dir, extract text page-by-page.
    Returns a list of dicts: {filename, title, full_text, num_pages}
    """
    documents = []
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        # Derive a human-readable title from the filename
        title = pdf_path.stem
        title = re.sub(r"v\d+$", "", title)
        title = title.replace("-", " ").replace("_", " ")
        title = re.sub(r"\s+", " ", title).strip()

        documents.append({
            "filename": pdf_path.name,
            "title": title,
            "full_text": "\n\n".join(pages_text),
            "num_pages": len(pages_text),
        })

    return documents


# ── 2. Text Cleaning ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw PDF-extracted text:
    - Decode HTML entities (&amp;, &nbsp;, &#39;, etc.)
    - Strip any residual HTML/XML tags
    - Fix broken hyphenation from line wraps
    - Remove page headers/footers, page numbers, journal boilerplate
    - Remove DOI lines, copyright notices, watermarks
    - Normalize whitespace and collapse blank lines
    """
    # ── HTML artifact cleanup ──
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)

    # ── Hyphenation and line-break repair ──
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)

    # ── Page numbers and running headers/footers ──
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^.*Page\s+\d+\s*(of\s*\d+)?.*$", "",
        text, flags=re.MULTILINE | re.IGNORECASE,
    )

    # ── Journal / publisher boilerplate ──
    text = re.sub(
        r"^(Downloaded from|Published by|Available at|Authorized licensed use"
        r"|This content downloaded|All rights reserved|Provided by"
        r"|Accessed on|View publication stats|See discussions, stats"
        r"|Content courtesy of|Redistribution subject to).*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Journal footer lines (e.g., "Frontiers in Human Dynamics 01 frontiersin.org")
    text = re.sub(
        r"^.*(?:frontiersin\.org|Frontiers\s+in\s+\w+\s+\w+\s+\d+).*$",
        "", text, flags=re.MULTILINE,
    )

    # ── Figure/table artifact cleanup ──
    # Remove lines that are just axis labels, arrows, or figure fragments
    # (common pdfplumber artifacts from extracted charts/graphs)
    text = re.sub(r"^[^\w]*[↵↑↓←→⇤⇥▶◀▲▼]+[^\w]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Fig\.\s*\d+.*$", "", text, flags=re.MULTILINE)

    # DOI lines
    text = re.sub(
        r"^.*(?:DOI|doi)\s*[:.]?\s*10\.\d{4,}/\S+.*$", "",
        text, flags=re.MULTILINE,
    )
    text = re.sub(r"^.*https?://doi\.org/\S+.*$", "", text, flags=re.MULTILINE)

    # Copyright lines
    text = re.sub(
        r"^.*(?:\xa9|Copyright\s+\d{4}|All rights reserved).*$", "",
        text, flags=re.MULTILINE | re.IGNORECASE,
    )

    # "Cite this article" / "How to cite"
    text = re.sub(
        r"^.*(Cite this|How to cite|Citation:).*$", "",
        text, flags=re.MULTILINE | re.IGNORECASE,
    )

    # ISSN/ISBN lines
    text = re.sub(r"^.*(?:ISSN|ISBN)[\s:]+[\d\-X]+.*$", "", text, flags=re.MULTILINE)

    # Received/Accepted/Published date lines
    text = re.sub(
        r"^.*(?:Received|Accepted|Published|Revised)\s*:?\s*\d{1,2}\s+\w+\s+\d{4}.*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE,
    )

    # ── Unicode whitespace normalization ──
    text = text.replace("\xa0", " ")
    text = text.replace(" ", " ")
    text = text.replace("​", "")
    text = text.replace("﻿", "")
    text = text.replace("‎", "")
    text = text.replace("‏", "")
    text = text.replace("\xad", "")

    # ── Final whitespace cleanup ──
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── 3. Chunking ───────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` characters
    with `overlap` characters of overlap between consecutive chunks.

    Strategy:
      1. Split on paragraph boundaries (double newlines).
      2. Greedily accumulate paragraphs until adding the next would
         exceed chunk_size.
      3. If a single paragraph exceeds chunk_size, fall back to
         splitting it by sentences, then by hard character cuts.
      4. Apply overlap by carrying trailing characters from the
         previous chunk into the next.
    """
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk = ""
    carry_over = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                carry_over = current_chunk.strip()[-overlap:]
                current_chunk = ""

            sub_chunks = _split_long_paragraph(para, chunk_size, overlap)
            for i, sc in enumerate(sub_chunks):
                if i == 0 and carry_over:
                    sc = carry_over + "\n\n" + sc
                    if len(sc) > chunk_size + overlap:
                        sc = sc[-(chunk_size):]
                chunks.append(sc.strip())
            carry_over = sub_chunks[-1].strip()[-overlap:] if sub_chunks else ""
            continue

        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                carry_over = current_chunk.strip()[-overlap:]

            if carry_over:
                current_chunk = carry_over + "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out tiny chunks (< 100 chars) that are likely figure/table artifacts
    MIN_CHUNK_CHARS = 100
    chunks = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

    return chunks


def _split_long_paragraph(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split an oversized paragraph by sentences first,
    then by hard character boundaries as a last resort.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) > 1:
        sub_chunks = []
        current = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    sub_chunks.append(current)
                current = sent
        if current:
            sub_chunks.append(current)

        if len(sub_chunks) > 1:
            overlapped = [sub_chunks[0]]
            for i in range(1, len(sub_chunks)):
                prev_tail = sub_chunks[i - 1][-overlap:]
                overlapped.append(prev_tail + " " + sub_chunks[i])
            return overlapped
        return sub_chunks

    sub_chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        sub_chunks.append(text[start:end])
        start = end - overlap
    return sub_chunks


# ── 4. Main Pipeline ─────────────────────────────────────────────

def run_pipeline(
    docs_dir: Path = DOCUMENTS_DIR,
    output_dir: Path = OUTPUT_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Full pipeline: ingest -> save raw -> clean -> chunk -> save.
    Returns all chunks with metadata.
    """
    print("=" * 60)
    print("Document Ingestion & Chunking Pipeline")
    print("=" * 60)
    print(f"  Chunk size:  {chunk_size} chars (~{chunk_size // 4} tokens)")
    print(f"  Overlap:     {overlap} chars (~{overlap // 4} tokens)")
    print(f"  Source dir:  {docs_dir}")
    print()

    # Step 1: Ingest
    print("[1/4] Ingesting documents...")
    documents = ingest_documents(docs_dir)
    print(f"  Loaded {len(documents)} documents\n")

    # Step 2: Save raw text (before any cleaning)
    raw_dir = docs_dir.parent / "raw_text"
    raw_dir.mkdir(exist_ok=True)
    print("[2/4] Saving raw text...")
    for doc in documents:
        raw_path = raw_dir / (Path(doc["filename"]).stem + ".txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(doc["full_text"])
        print(f"  {raw_path.name[:50]:50s}  {len(doc['full_text']):>8,} chars")
    print()

    # Step 3: Clean
    print("[3/4] Cleaning text...")
    for doc in documents:
        raw_len = len(doc["full_text"])
        doc["full_text"] = clean_text(doc["full_text"])
        cleaned_len = len(doc["full_text"])
        print(f"  {doc['filename'][:50]:50s}  {raw_len:>8,} -> {cleaned_len:>8,} chars")
    print()

    # Step 4: Chunk
    print("[4/4] Chunking documents...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["full_text"], chunk_size, overlap)
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['filename']}::chunk_{i}",
                "source_file": doc["filename"],
                "source_title": doc["title"],
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "text": chunk,
                "char_count": len(chunk),
            })
        print(f"  {doc['filename'][:50]:50s}  -> {len(doc_chunks):>4} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # Stats
    char_counts = [c["char_count"] for c in all_chunks]
    print(f"  Avg chunk size: {sum(char_counts) / len(char_counts):.0f} chars")
    print(f"  Min chunk size: {min(char_counts)} chars")
    print(f"  Max chunk size: {max(char_counts)} chars")

    oversized = sum(1 for c in char_counts if c > chunk_size + overlap)
    if oversized:
        print(f"  WARNING: {oversized} chunks exceed {chunk_size + overlap} chars")

    # Save
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_file}")
    print("=" * 60)

    return all_chunks


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk documents")
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help="Chunk size in characters (default: 2000)",
    )
    parser.add_argument(
        "--overlap", type=int, default=CHUNK_OVERLAP,
        help="Overlap in characters (default: 500)",
    )
    parser.add_argument(
        "--docs-dir", type=str, default=str(DOCUMENTS_DIR),
        help="Path to documents directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Path to output directory",
    )
    args = parser.parse_args()

    run_pipeline(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
