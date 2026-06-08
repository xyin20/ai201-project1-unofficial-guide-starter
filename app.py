"""
Gradio Web Interface
=====================
RAG query interface for the Human-AI Mutual Adaptation knowledge base.

Usage:
    python app.py
    # Opens at http://localhost:7860
"""

import gradio as gr
from query import ask


def handle_query(question: str) -> tuple[str, str]:
    """Process a query and return formatted answer + sources."""
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = ask(question)
    except ValueError as e:
        return f"Configuration error: {e}", ""
    except Exception as e:
        return f"Error: {e}", ""

    # Format sources — programmatically guaranteed, not LLM-dependent
    source_lines = []
    for src in result["sources_retrieved"]:
        # Mark which ones the LLM actually cited
        marker = " (cited)" if src in result["sources_cited"] else ""
        source_lines.append(f"  {src}{marker}")

    sources_text = (
        f"Retrieved {result['chunks_used']} chunks from:\n"
        + "\n".join(source_lines)
        + f"\n\nBest match distance: {min(result['distances']):.4f}"
    )

    return result["answer"], sources_text


# ── Build the Interface ──────────────────────────────────────────

with gr.Blocks(
    title="Human-AI Adaptation Research Assistant",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        "## Human-AI Mutual Adaptation Research Assistant\n"
        "Ask questions about human-AI/robot mutual adaptation, "
        "complex adaptive systems, and co-learning. "
        "Answers are grounded in 13 academic papers."
    )

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(
                label="Your question",
                placeholder="e.g., What is the Bounded-Memory Adaptation Model?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            btn = gr.Button("Ask", variant="primary", size="lg")

    answer = gr.Textbox(label="Answer", lines=10, interactive=False)
    sources = gr.Textbox(label="Sources", lines=5, interactive=False)

    # Example queries from the evaluation plan
    gr.Examples(
        examples=[
            "What is the Bounded-Memory Adaptation Model (BAM) and how does it enable mutual adaptation?",
            "What are the three classes of latent dynamics models used in human-robot mutual adaptation?",
            "How does correlation neglect affect human-AI collaboration in the Bayesian framework?",
            "What are the five requirements for successful human-robot co-learning?",
            "How does the system-theoretical approach characterize human interaction with agentic AI?",
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
