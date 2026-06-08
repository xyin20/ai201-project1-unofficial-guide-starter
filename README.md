# The Unofficial Guide — Project 1

## Domain

Human-AI and human-robot mutual adaptation, examined through complex adaptive systems (CAS) theory and dynamic information theory. This knowledge base covers 13 academic papers spanning HRI, cognitive science, decision theory, and systems theory. The domain is valuable because interdisciplinary researchers face severe terminology fragmentation — the same concept appears as "co-adaptation" in HRI, "mutual learning" in cognitive science, "complementarity" in decision theory, and "shared autonomy" in robotics. A RAG system that retrieves across these vocabularies saves researchers from manually cross-referencing papers across disciplinary boundaries.

---

## Document Sources


| #  | Source                                                                                           | Type            | URL or file path                                                                             |
| -- | ------------------------------------------------------------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------- |
| 1  | Mapping Human-Agent Co-Learning and Co-Adaptation: A Scoping Review                              | PDF (arXiv)     | https://arxiv.org/pdf/2506.06324                                                             |
| 2  | Mutual Adaptation and Influence: Survey of Latent Dynamics Models in HRI                         | PDF (TechRxiv)  | https://www.techrxiv.org/users/951609/articles/1320945                                       |
| 3  | Deconstructing Human-AI Collaboration: Agency, Interaction, and Adaptation                       | PDF (arXiv)     | https://arxiv.org/pdf/2404.12056                                                             |
| 4  | Human-Robot Mutual Adaptation in Collaborative Tasks (Nikolaidis et al.)                         | PDF (SAGE)      | https://journals.sagepub.com/doi/10.1177/0278364917690593                                    |
| 5  | Formalizing Human-Robot Mutual Adaptation: A Bounded Memory Model                                | PDF (CMU)       | https://personalrobotics.cs.washington.edu/publications/nikolaidis2016bam.pdf                |
| 6  | Mathematical Models of Adaptation in Human-Robot Collaboration                                   | PDF (arXiv)     | https://arxiv.org/pdf/1707.02586                                                             |
| 7  | Human-Robot Mutual Adaptation in Shared Autonomy                                                 | PDF (PMC)       | https://pmc.ncbi.nlm.nih.gov/articles/PMC6563347/                                            |
| 8  | A Bayesian Framework for Human-AI Collaboration                                                  | PDF (arXiv)     | https://arxiv.org/pdf/2602.14331                                                             |
| 9  | Explaining and Improving Information Complementarities in Multi-Agent Decision-Making            | PDF (arXiv)     | https://arxiv.org/pdf/2502.06152                                                             |
| 10 | Human-Artificial Interaction in the Age of Agentic AI: A System-Theoretical Approach             | PDF (Frontiers) | https://www.frontiersin.org/journals/human-dynamics/articles/10.3389/fhumd.2025.1579166/full |
| 11 | The Role of Adaptation in Collective Human-AI Teaming                                            | PDF (PubMed)    | https://pubmed.ncbi.nlm.nih.gov/36374986/                                                    |
| 12 | Shifting the Human-AI Relationship: Toward a Dynamic Relational Learning-Partner Model           | PDF (arXiv)     | https://arxiv.org/pdf/2410.11864                                                             |
| 13 | Adaptation Through Communication: Assessing Human-AI Partnership for Complex Engineering Systems | PDF (ASME)      | https://asmedigitalcollection.asme.org/mechanicaldesign/article/146/8/081401/1193985/        |

---

## Chunking Strategy

**Chunk size:** ~2000 characters (~512 tokens)

**Overlap:** ~500 characters (~128 tokens), representing 25% of chunk size

**Why these choices fit your documents:** Academic papers contain dense, multi-sentence arguments where a single concept often spans an entire paragraph. A 2000-character chunk is large enough to capture a complete paragraph with its surrounding context (e.g., a model definition together with its notation and motivation), but small enough to avoid mixing unrelated sections like methods bleeding into results. The 25% overlap ensures that concepts spanning paragraph boundaries are captured in at least one chunk. The chunking algorithm splits on paragraph boundaries first (double newlines), falling back to sentence-level splits and then hard character cuts for oversized paragraphs. A minimum 100-character filter removes tiny garbage chunks from extracted figure/chart data.

**Preprocessing:** Text extraction via pdfplumber, followed by cleaning that decodes HTML entities, strips residual HTML tags, repairs line-break hyphenation, removes page numbers, journal boilerplate (DOI lines, copyright notices, "Cite this article" blocks, Frontiers footers, ISSN/ISBN lines, Received/Accepted date lines), and normalizes unicode whitespace. Raw text is saved to `raw_text/` before any cleaning.

**Final chunk count:** 593 chunks across 13 documents (avg 1823 chars, min 132, max 2499)

---

## Embedding Model

**Model used:** SPECTER2 (`allenai/specter2` proximity adapter on `allenai/specter2_base`) via the `adapters` library. SPECTER2 was trained by the Allen Institute for AI on 6M+ citation triplets from scientific papers using contrastive learning. Its 512-token input window matches the chunk size exactly.

**Production tradeoff reflection:** SPECTER2 was chosen because all 13 sources are academic publications. It encodes domain-specific terminology like "POMDP," "co-adaptation," "shared autonomy," and "latent dynamics" more accurately than general-purpose models like `all-MiniLM-L6-v2`. In a production deployment, the main tradeoffs would be: (1) context length — SPECTER2's 512-token limit constrains chunk size, whereas models like `nomic-embed-text` support 8192 tokens; (2) latency — SPECTER2 runs locally without API calls, which is good for privacy but slower than API-hosted models on GPU infrastructure; (3) cross-lingual support — SPECTER2 is English-only, which would matter if the corpus included non-English papers; (4) domain specificity vs. generality — a general-purpose model would handle non-academic queries better, but would lose the scientific vocabulary precision that makes SPECTER2 effective here.

---

## Grounded Generation

**System prompt grounding instruction:** The system prompt uses imperative language with 5 numbered rules under a "STRICT RULES" header. The critical enforcement mechanisms are:

1. Rule 1 explicitly prohibits the LLM from using its own training knowledge: "ONLY use information from the DOCUMENTS provided below. Do NOT use your own training knowledge, background assumptions, or general information about the topic."
2. Rule 2 provides an exact decline phrase for out-of-scope queries: "respond EXACTLY with: 'I don't have enough information in the provided documents to answer that question.'"
3. Rule 3 requires `[SOURCE N]` citations in every answer.
4. Temperature is set to 0.1 to minimize creative invention.

The context is formatted with labeled sections (`[SOURCE 1] (from: filename.pdf)`) so the LLM can reference specific documents and we can programmatically verify which sources it cited.

**How source attribution is surfaced in the response:** Source attribution is programmatically guaranteed through two independent mechanisms. First, the LLM is instructed to cite `[SOURCE N]` labels, and `_extract_cited_sources()` parses these references from the response to determine which files the LLM actually cited. Second, regardless of what the LLM writes, the system always returns `sources_retrieved` — the full list of source filenames from retrieval metadata. The Gradio UI displays both, marking LLM-cited sources with "(cited)" so the user can see exactly which documents contributed to the answer.

---

## Evaluation Report


| # | Question                                                                                                                             | Expected answer                                                                                                                                                                                                                                               | System response (summarized)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Retrieval quality | Response accuracy  |
| - | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------ |
| 1 | What is the Bounded-Memory Adaptation Model (BAM) and how does it enable mutual adaptation?                                          | BAM models the human as having bounded memory over recent interactions, integrated into a POMDP. The robot infers whether the human is adaptable — if so, it guides them toward the optimal strategy; otherwise, it aligns with the human to maintain trust. | BAM is based on "bounded rationality," modeling the human as making decisions on the last k interactions. It enables mutual adaptation via a parameterization of the transition function based on human adaptability α. Cited nikolaidis2016bam.pdf and nihms-1032728.pdf. Distances: 0.064–0.069.                                                                                                                                                                                                                                                                                                                                                                                          | Relevant          | Accurate           |
| 2 | What are the three classes of latent dynamics models used in human-robot mutual adaptation?                                          | Bayesian, Markovian, and encoded models, forming a unified framework with prediction and control steps.                                                                                                                                                       | the three classes of latent dynamics models used in human-robot mutual adaptation are: Bayesian, Markovian, and encoded. Cited: Mutual Adaptation and Influence Survey of Latent Dynamics Models in HRI.pdf Distance: 0.0447-0.054                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Relevant          | Accurate           |
| 3 | How does correlation neglect affect human-AI collaboration in the Bayesian framework?                                                | Humans treat partially overlapping AI signals as independent, over-weighting AI recommendations when overlap is high, reducing complementarity.                                                                                                               | Correlation neglect occurs when humans treat partially overlapping signals as conditionally independent, leading to double-counting of shared evidence. Weak AI can strictly impair performance due to this double-counting, creating an "impairment region." Cited 2602.14331v1.pdf. Distances: 0.044–0.063.                                                                                                                                                                                                                                                                                                                                                                                | Relevant          | Accurate           |
| 4 | What are the five requirements for successful human-robot co-learning?                                                               | Shared goal, synchrony, interdependence, adaptability, and transparency.                                                                                                                                                                                      | I don't have enough information in the provided documents to answer that question.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Off-target        | Inaccurate         |
| 5 | How does the system-theoretical approach characterize human interaction with agentic AI differently from traditional HCI frameworks? | It frames interaction as a dynamic system with feedback loops and emergent behavior rather than a static user-interface model.                                                                                                                                | the system-theoretical approach characterizes human interaction with agentic AI as a dynamic interplay between human and computational agents within a networked system, emphasizing coordination and communication among heterogeneous agents. This approach goes beyond traditional interface-based approaches, distinguishing between Multi-Agent Systems (MAS) and Centaurian systems, which integrate human and AI capabilities for unified decision making. [SOURCE 1] also notes that this perspective frames HCI as a dynamic interplay between human and computational agents, rather than a traditional interface-based approach. Cited: fhumd-7-1579166.pdf Distance: 0.058-0.0788 | Off-target        | Partially accurate |

**Retrieval quality scale:** Relevant / Partially relevant / Off-target
**Response accuracy scale:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed (partial):** "What are the five requirements for successful human-robot co-learning?"

**What the system returned:** The system responded with "I don't have enough information in the provided documents to answer that question." — a complete refusal. The expected answer (shared goal, synchrony, interdependence, adaptability, and transparency) is present in the scoping review paper (source #1, Mapping Human-Agent Co-Learning and Co-Adaptation), which is in the corpus.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval-stage failure** caused by vocabulary mismatch between the query and the chunk text. The five requirements are listed in the scoping review, but the paper uses the phrase "five themes" or "five key factors" rather than "five requirements." SPECTER2 embeds based on scientific terminology learned from citation context — the word "requirements" has a different semantic profile than "themes" or "factors" in scientific literature. As a result, the top-5 retrieved chunks likely came from papers discussing co-learning broadly (definitions, frameworks) rather than the specific chunk enumerating the five items. When none of the retrieved chunks contained the enumerated list, the generation stage correctly followed its grounding rule and declined to answer — the generation stage worked exactly as designed, but it was given the wrong context by retrieval.

A secondary contributing factor is **terminology fragmentation across disciplines**, which was identified as an anticipated challenge in planning.md. The query uses HRI-specific framing ("co-learning requirements"), but the scoping review draws on multiple fields and may frame the same concepts using different vocabulary, causing the relevant chunk to embed far from the query in vector space.

**What you would change to fix it:** Three approaches: (1) **Query expansion** — before embedding the query, append synonyms or rephrasings (e.g., "co-learning requirements OR themes OR factors OR conditions"). This directly addresses the vocabulary mismatch. (2) **Hybrid retrieval** — combine SPECTER2's semantic search with BM25 keyword search. BM25 would catch the chunk containing "co-learning" and "five" even if the semantic embedding misses. (3) **Increase top-k to 8–10** — retrieving more chunks increases the chance of capturing the relevant chunk even when it doesn't rank in the top 5, at the cost of diluting context with loosely related material.(1) Increase chunk size to ~3000 characters for this paper, which would capture both the formal model and its behavioral explanation in a single chunk; or (2) increase top-k to 7–8 to retrieve more chunks from the primary source. Alternatively, a re-ranking step after initial retrieval could prioritize multiple chunks from the same document when the query clearly targets a specific model by name.

---

## Spec Reflection

**One way the spec helped you during implementation:** The chunking strategy section of planning.md saved significant iteration time by pre-specifying the chunk size (2000 chars), overlap (500 chars), and the rationale for those numbers. When implementing `chunk_text()`, these concrete targets meant the function could be written, tested, and validated against clear criteria — "does the average chunk stay near 2000 chars, do boundary cases preserve context?" — rather than guessing at parameters and tuning blindly. The spec's note about keeping equation blocks with their surrounding explanations also directly informed the paragraph-boundary-first splitting strategy, which was more effective than a naive character-count split.

**One way your implementation diverged from the spec, and why:** The architecture diagram specified PyMuPDF (fitz) for PDF text extraction, but the implementation uses pdfplumber instead. PyMuPDF was unavailable in the development environment, and pdfplumber was already listed as an alternative in `requirements.txt`. Both libraries extract text at the same fidelity for these academic PDFs. Additionally, the spec listed `sentence-transformers` as the library for loading SPECTER2, but SPECTER2 actually requires the `adapters` library built on top of `transformers` — it uses an adapter architecture that sentence-transformers doesn't natively support. This was discovered during implementation by checking the SPECTER2 HuggingFace documentation, and the requirements.txt was updated accordingly.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* I gave Claude my Chunking Strategy section and Document Sources table from planning.md, and asked it to implement `ingest_documents()` to load PDFs and `chunk_text()` with 512-token chunks and 128-token overlap, splitting on paragraph boundaries first.
- *What it produced:* It returned a complete `ingest_and_chunk.py` with PDF extraction via pdfplumber, a `clean_text()` function for hyphenation repair and whitespace normalization, and a `chunk_text()` function using paragraph-boundary splitting with sentence-level and hard-character fallbacks. It also generated a `run_pipeline()` that saved raw text before cleaning and output a `chunks.json` with metadata.
- *What I changed or overrode:* The initial `clean_text()` function only handled basic whitespace normalization. After inspecting cleaned documents and sample chunks, I directed Claude to harden the cleaner by adding HTML entity decoding, journal boilerplate removal (Frontiers footers, DOI lines, copyright notices, ISSN/ISBN), figure/chart artifact stripping, and a minimum 100-character chunk filter to drop extracted graph data that produced nonsensical tiny chunks.

**Instance 2**

- *What I gave the AI:* I gave Claude my Retrieval Approach section from planning.md and asked it to implement `embed_and_retrieve.py` using `allenai/specter2` via `sentence-transformers`, storing embeddings in ChromaDB with source metadata, and a `retrieve(query, k=5)` function.
- *What it produced:* It initially generated code using `SentenceTransformer("allenai/specter2")` from the sentence-transformers library, which was incorrect. It also used `except ValueError` for ChromaDB's collection-not-found case, which raises `NotFoundError` in current versions.
- *What I changed or overrode:* After checking the SPECTER2 HuggingFace page, I directed Claude to rewrite the embedding code using the `adapters` library with `AutoAdapterModel.from_pretrained("allenai/specter2_base")` and `model.load_adapter("allenai/specter2")`. I also had it add explicit adapter activation (`model.set_active_adapters()`) after the first run showed the warning "none are activated for the forward pass," and fix the ChromaDB exception handling to catch `NotFoundError`.

**Instance 3**

- *What I gave the AI:* I asked Claude to generate the generation pipeline (`query.py`) and Gradio interface (`app.py`) based on the planning.md Architecture section, specifying Groq's `llama-3.3-70b-versatile`, grounding enforcement, and programmatic source attribution.
- *What it produced:* It generated a system prompt with 5 strict rules, a `_build_context()` function that labels each chunk as `[SOURCE N]`, a `_extract_cited_sources()` function that parses citations from the LLM's response, and a Gradio Blocks interface with example queries from the evaluation plan.
- *What I changed or overrode:* I reviewed the system prompt to confirm it enforces grounding rather than merely suggesting it — specifically checking that it uses prohibitive language ("Do NOT use your own training knowledge") rather than permissive language ("Try to use the documents"). I also verified that source attribution was programmatically guaranteed (via `sources_retrieved` from retrieval metadata) rather than solely dependent on the LLM remembering to cite sources.
