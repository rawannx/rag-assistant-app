"""Loads the persisted vector store and retrieves relevant chunks for a query."""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Wraps the embedding model + Chroma collection, loaded once at startup."""

    def __init__(self) -> None:
        logger.info("Loading embedding model '%s'...", settings.embedding_model)
        self.embedding_model = SentenceTransformer(settings.embedding_model)

        logger.info("Connecting to vector store at '%s'...", settings.vector_store_path)
        self.client = chromadb.PersistentClient(path=settings.vector_store_path)
        self.collection = self.client.get_collection(name=settings.collection_name)

        logger.info(
            "Retrieval service ready. Collection '%s' has %d chunks.",
            settings.collection_name,
            self.collection.count(),
        )

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        """Return the top-k most relevant chunks for a question."""
        k = k or settings.top_k
        query_embedding = self.embedding_model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "distance": dist,
                }
            )
        return retrieved
