"""API routes: GET /health and POST /query."""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health_check(request: Request) -> dict:
    """Simple liveness check that also confirms the vector store is loaded."""
    retrieval_service = request.app.state.retrieval_service
    return {
        "status": "ok",
        "indexed_chunks": retrieval_service.collection.count(),
    }


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve relevant context, ask the LLM, and return a grounded, cited answer."""
    retrieval_service = request.app.state.retrieval_service

    retrieved_chunks = retrieval_service.retrieve(payload.question)
    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    try:
        answer = generate_answer(payload.question, retrieved_chunks)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail="LLM generation failed.") from exc

    sources = sorted({f"{c['source']} (p.{c['page']})" for c in retrieved_chunks})
    return QueryResponse(answer=answer, sources=sources)
