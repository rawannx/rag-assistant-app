"""Application configuration, loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # Vector store
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "ml_rag_assistant"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 6

    # CORS
    frontend_origin: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
