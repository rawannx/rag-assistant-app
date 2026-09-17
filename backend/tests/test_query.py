"""Tests for the /query endpoint using FastAPI's TestClient.

We patch RetrievalService before the app starts up, so tests run fast and don't
require a real vector store, embedding model download, or a running Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    fake_retrieval_service = MagicMock()
    fake_retrieval_service.retrieve.return_value = [
        {"text": "K-means is a clustering algorithm.", "source": "clustering.pdf", "page": 2}
    ]
    fake_retrieval_service.collection.count.return_value = 42

    with patch("app.main.RetrievalService", return_value=fake_retrieval_service):
        from app.main import app  # imported here so the patch applies before startup

        with TestClient(app) as test_client:
            yield test_client


def test_query_happy_path(client):
    with patch(
        "app.api.routes.query.generate_answer",
        return_value="K-means groups data into k clusters.",
    ):
        response = client.post("/query", json={"question": "What is K-means?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["sources"] == ["clustering.pdf (p.2)"]


def test_query_invalid_input_returns_422(client):
    # Missing the required "question" field entirely
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
