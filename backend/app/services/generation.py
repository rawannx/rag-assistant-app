"""Builds the grounded prompt and calls the local Ollama LLM to generate an answer."""

import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a helpful Machine Learning study assistant.
Answer the question using ONLY the context below. If the context does not contain
the answer, say you don't have enough information instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in retrieved_chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Call the local Ollama model with the grounded prompt and return its answer."""
    prompt = build_prompt(question, retrieved_chunks)
    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception:
        logger.exception("Failed to generate answer from Ollama")
        raise
