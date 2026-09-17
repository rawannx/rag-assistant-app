"""FastAPI application entrypoint: creates the app, wires CORS, and loads the
retrieval service once at startup via a lifespan context manager."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import RetrievalService
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading vector store and embedding model...")
    app.state.retrieval_service = RetrievalService()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ML RAG Assistant API",
    description="RAG-powered API answering Machine Learning questions grounded in "
    "course lecture notes, scikit-learn docs, and pandas docs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/")
def root() -> dict:
    return {"message": "ML RAG Assistant API is running. See /docs for the API reference."}
