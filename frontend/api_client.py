"""Thin wrapper around the backend API so app.py doesn't deal with HTTP directly."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class APIError(Exception):
    """Raised when the backend can't be reached or returns an error."""


def ask_question(question: str, timeout: int = 60) -> dict:
    """Send a question to the backend /query endpoint.

    Returns a dict with 'answer' and 'sources'. Raises APIError on failure.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise APIError(
            "Can't reach the backend. Make sure the FastAPI server is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIError("The backend took too long to respond. Please try again.") from exc
    except requests.exceptions.HTTPError as exc:
        raise APIError(f"Backend returned an error: {exc.response.status_code}") from exc


def check_health() -> dict | None:
    """Return backend health info, or None if unreachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
