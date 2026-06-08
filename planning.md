# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Human-AI/robot mutual adaptation, examined through the lenses of complex adaptive systems (CAS) theory and dynamic information theory. Literature research is always a pain, especially for people who's doing interdisciplinary research like me, it's always challenge to understand what other discipline's paper talking about, also...even though I like doing research, I hate readings.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->


| #  | Source                                                                                            | Description                                                                                                                           | URL or location                                                                                                                                    |
| -- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Mapping Human-Agent Co-Learning and Co-Adaptation: A Scoping Review                               | Scoping review of co-learning and mutual adaptation across Web of Science, Engineering Village, and EBSCOhost (pre-2024)              | https://arxiv.org/pdf/2506.06324                                                                                                                   |
| 2  | Mutual Adaptation and Influence: Survey of Latent Dynamics Models in HRI                          | Survey identifying three classes of latent dynamics models (Bayesian, Markovian, encoded) with a unified prediction-control framework | https://www.techrxiv.org/users/951609/articles/1320945-mutual-adaptation-and-influence-survey-of-latent-dynamics-models-in-human-robot-interaction |
| 3  | Deconstructing Human-AI Collaboration: Agency, Interaction, and Adaptation                        | Framework paper on co-adaptation where both humans and AI learn from each other                                                       | https://arxiv.org/pdf/2404.12056                                                                                                                   |
| 4  | Human-Robot Mutual Adaptation in Collaborative Tasks: Models and Experiments (Nikolaidis et al.)  | Introduces the Bounded-Memory Adaptation Model (BAM) integrated into a POMDP for mutual adaptation                                    | https://journals.sagepub.com/doi/10.1177/0278364917690593                                                                                          |
| 5  | Formalizing Human-Robot Mutual Adaptation: A Bounded Memory Model                                 | Foundational formalism for bounded-memory human adaptation modeling                                                                   | https://personalrobotics.cs.washington.edu/publications/nikolaidis2016bam.pdf                                                                      |
| 6  | Mathematical Models of Adaptation in Human-Robot Collaboration                                    | Mutual adaptation formalism where the robot infers human characteristics and adapts its actions                                       | https://arxiv.org/pdf/1707.02586                                                                                                                   |
| 7  | Human-Robot Mutual Adaptation in Shared Autonomy                                                  | Explores mutual adaptation under shared control paradigms                                                                             | https://pmc.ncbi.nlm.nih.gov/articles/PMC6563347/                                                                                                  |
| 8  | A Bayesian Framework for Human-AI Collaboration: Complementarity and Correlation Neglect          | Information-theoretic model of how humans integrate AI signals with own information                                                   | https://arxiv.org/pdf/2602.14331                                                                                                                   |
| 9  | EXPLAINING AND IMPROVING INFORMATION COMPLEMENTARITIES IN MULTI-AGENT DECISION-MAKING             | Statistical decision theory approach analyzing information value across human-alone, AI-alone, and team conditions                    | [2502.06152v6.pdf](https://arxiv.org/pdf/2502.06152)                                                                                               |
| 10 | Human-Artificial Interaction in the Age of Agentic AI: A System-Theoretical Approach              | Systems theory framework for human-AI interaction with agentic AI                                                                     | https://www.frontiersin.org/journals/human-dynamics/articles/10.3389/fhumd.2025.1579166/full                                                       |
| 11 | The Role of Adaptation in Collective Human–AI Teaming                                            | Framework for adaptive AI in group settings; technical challenges for collaborative AI                                                | https://pubmed.ncbi.nlm.nih.gov/36374986/                                                                                                          |
| 12 | Shifting the Human-AI Relationship: Toward a Dynamic Relational Learning-Partner Model            | Proposes viewing human-AI interaction as a cooperative, evolving partnership                                                          | https://arxiv.org/pdf/2410.11864                                                                                                                   |
| 13 | Adaptation Through Communication: Assessing Human–AI Partnership for Complex Engineering Systems | Adaptation mechanisms through communication in engineering design contexts                                                            | https://asmedigitalcollection.asme.org/mechanicaldesign/article/146/8/081401/1193985/                                                              |

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size: \~512 tokens (\~2000 characters)**

**Overlap: 128 tokens (\~500 characters)**

**Reasoning: A 512-token chunk is large enough to capture a complete paragraph or a key definition with its surrounding context, but small enough to avoid mixing unrelated sections (e.g., a methods paragraph bleeding into results). The 25% overlap ensures that concepts spanning paragraph boundaries. Since most of research papers have multiple paragraphs and the paragraphs are very long, plus the questions like "How does the BAM model handle adaptable vs. non-adaptable humans?" needs a chunk containing both the model name *and* the branching logic, a 2000 characters chunk would be a reasonable chunk, as too large of the chunk will also causing problems like covering both "methods" and "results", which would be probmatic.**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model: `specter2` via `allenai`**

**Top-k: 5, I'd like to have some right papers plus some supporting context from related work**

**Production tradeoff reflection: This model was trained by the Allen Institute for AI specifically on scientific papers, using citation-based contrastive learning. Since all 13 sources are academic publications, SPECTER2 should encode domain-specific terminology like "POMDP," "co-adaptation," "shared autonomy," "latent dynamics" more accurately than general-purpose models like `all-MiniLM-L6-v2`.**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->


| # | Question                                                                                                                             | Expected answer                                                                                                                                                                                                                                                                                                                                   |
| - | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | What is the Bounded-Memory Adaptation Model (BAM) and how does it enable mutual adaptation?                                          | BAM is a computational model where the robot models the human as having bounded memory over recent interactions, integrated into a POMDP. The robot infers whether the human is adaptable — if so, the robot may disagree expecting the human to switch to the optimal strategy; otherwise, it aligns with the human's policy to maintain trust. |
| 2 | What are the three classes of latent dynamics models used in human-robot mutual adaptation?                                          | Bayesian, Markovian, and encoded models. They form a unified framework with a prediction step (using latent dynamics to anticipate human behavior) and a control step (using latent dynamics to influence the human).                                                                                                                             |
| 3 | How does correlation neglect affect human-AI collaboration in the Bayesian framework?                                                | Humans tend to treat partially overlapping information signals from AI as conditionally independent, leading them to over-weight AI recommendations when the AI's information substantially overlaps with their own, reducing effective complementarity.                                                                                          |
| 4 | What are the five requirements for successful human-robot co-learning?                                                               | Shared goal, synchrony, interdependence, adaptability, and transparency.                                                                                                                                                                                                                                                                          |
| 5 | How does the system-theoretical approach characterize human interaction with agentic AI differently from traditional HCI frameworks? | It frames the interaction as a dynamic system with feedback loops and emergent behavior rather than a static user-interface model, emphasizing that both agents continuously modify each other's state through interaction, consistent with complex adaptive systems theory.                                                                      |

---

## Anticipated Challenges

**Terminology fragmentation across fields. **The same concept appears under different names — "co-adaptation" (HRI), "mutual learning" (cognitive science), "complementarity" (decision theory), "shared autonomy" (robotics). Chunks from different papers discussing the same phenomenon may embed far apart in vector space because they use different vocabulary, causing relevant results to be missed during retrieval. Mitigation: consider query expansion or synonym mapping at query time.
**Dense mathematical notation breaking chunk coherence.** Several papers (#4–6, #8) contain heavy mathematical formalism (POMDPs, Bayesian inference equations). These sections don't embed meaningfully as text — an equation block without its surrounding prose explanation is useless, but including too much context inflates chunk size. Chunks that split a theorem from its proof or an equation from its variable definitions will produce poor retrieval results.

---

## Architecture

┌─────────────────────┐
│  Document Ingestion  │  Tool: PyMuPDF (fitz) for PDFs, requests + BeautifulSoup for HTML
│  (13 academic papers)│  Output: raw text per document
└────────┬────────────┘
│
▼
┌─────────────────────┐
│     Chunking         │  Tool: custom Python (chunk_text())
│  512 tokens, 128     │  Strategy: split on paragraph/section boundaries,
│  token overlap       │  fall back to token-count splits with overlap
└────────┬────────────┘
│
▼
┌─────────────────────┐
│  Embedding +         │  Tool: sentence-transformers (allenai/specter2)
│  Vector Store        │  Store: ChromaDB (local, persistent)
│                      │  Metadata: source title, section, chunk index
└────────┬────────────┘
│
▼
┌─────────────────────┐
│     Retrieval        │  Tool: ChromaDB similarity search
│  top-k = 5           │  Input: user query → embed → cosine similarity
└────────┬────────────┘
│
▼
┌─────────────────────┐
│     Generation       │  Tool: OpenAI API (GPT-4o) or Claude API
│                      │  Prompt: system instructions + retrieved chunks + user query
│                      │  Interface: Streamlit or CLI
└─────────────────────┘

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:** I'll give Claude my Chunking Strategy section and document table, and ask it to implement `ingest_documents()` to download/parse each source (PDF via PyMuPDF, HTML via BeautifulSoup) and `chunk_text()` with 512-token chunks and 128-token overlap, splitting on paragraph boundaries first. I'll verify by checking that chunks from paper #4 (Nikolaidis BAM) keep equation blocks with their surrounding explanations, and that no chunk exceeds the specified size.

**Milestone 4 — Embedding and retrieval:** I'll give Claude my Retrieval Approach section and ask it to implement `embed_chunks()` using `sentence-transformers` with `allenai/specter2`, store embeddings in ChromaDB with source metadata (title, section, chunk index), and implement `retrieve(query, k=5)` using cosine similarity. Since SPECTER2 has a 512-token input window matching my chunk size, I'll verify that no chunks are truncated during embedding. I'll then run my 5 evaluation questions and check that the top-5 chunks come from the expected source papers (e.g., question #1 should retrieve chunks from papers #4 and #5). I'll also test whether SPECTER2's scientific training helps with cross-terminology retrieval — e.g., querying "how robots adjust to people" should still match chunks using "mutual adaptation in shared autonomy."

**Milestone 5 — Generation and interface:** I'll give Claude my Evaluation Plan and Architecture sections and ask it to implement `generate_answer(query, chunks)` with a system prompt instructing the LLM to answer only from provided context and cite sources. I'll build a Streamlit interface with a query box and source attribution display. I'll verify by running all 5 evaluation questions and comparing generated answers against expected answers, checking for hallucination and correct source citation.
