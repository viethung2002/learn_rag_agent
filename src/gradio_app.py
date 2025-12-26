import json
import logging
from typing import Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG"]


async def stream_response(
    query: str,
    top_k: int = 3,
    use_hybrid: bool = True,
    model: str = DEFAULT_MODEL,
    categories: str = "",
) -> Iterator[str]:
    """Stream response from Gemini RAG API (SSE)"""

    if not query.strip():
        yield "⚠️ Please enter a question."
        return

    # Parse categories
    category_list = (
        [c.strip() for c in categories.split(",") if c.strip()]
        if categories else None
    )

    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "categories": category_list,
    }

    url = f"{API_BASE_URL}/stream_gemini/stream"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:

                if response.status_code != 200:
                    yield f"❌ API Error: {response.status_code}"
                    return

                answer_buffer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    try:
                        data = json.loads(line.removeprefix("data:").strip())
                    except json.JSONDecodeError:
                        continue

                    # ❌ Backend error
                    if "error" in data:
                        yield f"❌ Error: {data['error']}"
                        return

                    # 🔹 Streaming token
                    if "chunk" in data:
                        answer_buffer += data["chunk"]
                        yield answer_buffer
                        continue

                    # 🔹 Final metadata
                    if data.get("done") is True:
                        answer_buffer = data.get("answer", answer_buffer)
                        sources = data.get("sources", [])
                        chunks_used = data.get("chunks_used", 0)
                        search_mode = data.get("search_mode", "rag")

                        final_output = answer_buffer

                        if sources or chunks_used:
                            final_output += "\n\n---\n**Search Info:**\n"
                            final_output += f"- Mode: {search_mode}\n"
                            final_output += f"- Chunks used: {chunks_used}\n"

                            if sources:
                                final_output += f"- Sources: {len(sources)} papers\n"
                                for i, src in enumerate(sources[:3], 1):
                                    name = src.split("/")[-1]
                                    final_output += f"  {i}. [{name}]({src})\n"
                                if len(sources) > 3:
                                    final_output += f"  ... and {len(sources) - 3} more\n"

                        yield final_output
                        return

    except httpx.RequestError as e:
        yield f"❌ Connection error: {e}"
    except Exception as e:
        yield f"❌ Unexpected error: {e}"


def create_gradio_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="arXiv Paper Curator - RAG Chat",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🔬 arXiv Paper Curator - RAG Chat
            
            Ask questions about machine learning and AI research papers from arXiv.
            The system will search through indexed papers and provide answers with sources.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question", placeholder="What are transformers in machine learning?", lines=2, max_lines=5
                )

            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="More chunks = more context but slower generation",
                    )

                    use_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better results than keyword-only search",
                    )

                    model_choice = gr.Dropdown(
                        choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                        value=DEFAULT_MODEL,
                        label="LLM Model",
                        info="Larger models may give better answers but are slower",
                    )

                    categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG, cs.CL",
                        info="Comma-separated. Leave empty for all categories",
                    )

        response_output = gr.Markdown(
            label="Answer", value="Ask a question to get started!", height=400, elem_classes=["response-markdown"]
        )

        # Examples
        gr.Examples(
            examples=[
                ["What are transformers in machine learning?", 3, True, "llama3.2:1b", "cs.AI, cs.LG"],
                ["How do convolutional neural networks work?", 5, True, "llama3.2:1b", "cs.CV, cs.LG"],
                ["What is attention mechanism in deep learning?", 4, False, "llama3.2:1b", "cs.AI"],
                ["Explain reinforcement learning algorithms", 3, True, "llama3.2:1b", "cs.LG, cs.AI"],
                ["What are the latest developments in NLP?", 5, True, "llama3.2:1b", "cs.CL"],
            ],
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
        )

        # Handle submission
        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        # Handle Enter key
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        gr.Markdown(
            """
            ---
            
            **Note**: Make sure the RAG API server is running at `http://localhost:8000` before using this interface.
            
            **Categories**: cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computational Linguistics), 
            cs.CV (Computer Vision), cs.NE (Neural Networks), stat.ML (Statistics - Machine Learning)
            """
        )

    return interface


def main():
    """Main entry point for the Gradio app"""
    print("🚀 Starting arXiv Paper Curator Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed to avoid port conflict
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
