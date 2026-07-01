from __future__ import annotations

import os
from typing import Sequence

import requests
from dotenv import load_dotenv


class OllamaEmbedder:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv(
            "EMBEDDING_MODEL", "qwen3-embedding:0.6b"
        )
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.timeout = timeout or float(os.getenv("EMBEDDING_TIMEOUT", "120"))

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [str(text).strip() for text in texts]
        if not inputs:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": inputs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise RuntimeError("Ollama returned an invalid embedding response.")
        return [[float(value) for value in vector] for vector in embeddings]
