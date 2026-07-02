"""Long-term semantic memory for Switchboard.

Memory items are embedded with a local Ollama embedding model
(``nomic-embed-text`` by default) and retrieved by cosine similarity. The
embedding model runs locally, so long-term memory never leaves the machine.
When the embedding model is unavailable, retrieval falls back to the existing
SQLite text search so behavior degrades gracefully.

Privacy: retrieved memories are injected into backend context as a
``<long_term_memory>`` block and pass through the same secret redaction as
recent conversation context. Memory retrieval can be disabled entirely via
configuration (``semantic_memory_enabled: false``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Literal

import httpx
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from switchboard.app.models.personal import PersonalMemoryRead
from switchboard.app.models.telemetry import MemoryEmbeddingRecord
from switchboard.app.storage.repositories import MemoryRepository


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the local embedding model cannot be reached."""


EmbeddingTask = Literal["classification", "search_document", "search_query"]
NOMIC_EMBED_CONTEXT = 8192


def _embedding_prompt(model: str, text: str, task: EmbeddingTask) -> str:
    model_name = model.lower()
    if model_name == "nomic-embed-text":
        return f"{task}: {text}"
    if model_name == "qwen3-embedding:0.6b":
        instructions = {
            "classification": "Represent this text for routing classification.",
            "search_document": "Represent this saved memory for retrieval.",
            "search_query": "Represent this query for retrieving relevant saved memories.",
        }
        return f"Instruct: {instructions[task]}\nQuery: {text}"
    return text


class OllamaEmbeddingClient:
    """Minimal synchronous client for Ollama's embedding endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout_s: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed(
        self,
        text: str,
        *,
        task: EmbeddingTask = "classification",
    ) -> list[float]:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": _embedding_prompt(self.model, text, task),
        }
        if self.model.lower() == "nomic-embed-text":
            payload["options"] = {"num_ctx": NOMIC_EMBED_CONTEXT}
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(f"Embedding model unreachable: {exc}") from exc
        embedding = response.json().get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingUnavailableError("Embedding model returned an empty vector.")
        return [float(value) for value in embedding]

    def embed_classification(self, text: str) -> list[float]:
        return self.embed(text, task="classification")

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text, task="search_document")

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text, task="search_query")


class CachedEmbedder:
    """Small LRU over an embed callable.

    Several learned components (router, tool dispatcher, semantic memory,
    sensitivity escalator) embed the same prompt during one request. Sharing
    a single cached embedder means the prompt hits the embedding model once,
    not once per component. Only successful embeddings are cached; errors
    propagate so each caller keeps its own fallback behavior.
    """

    def __init__(
        self,
        embed: Callable[[str], list[float]],
        *,
        maxsize: int = 64,
    ) -> None:
        self._embed_fn = embed
        self._maxsize = maxsize
        self._cache: dict[str, list[float]] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vector = self._embed_fn(text)
        self._cache[text] = vector
        self._order.append(text)
        while len(self._order) > self._maxsize:
            stale = self._order.pop(0)
            self._cache.pop(stale, None)
        return vector


class MemoryEmbeddingRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def upsert(
        self,
        *,
        memory_id: int,
        project: str,
        embedding_model: str,
        vector: list[float],
    ) -> None:
        with Session(self.engine) as session:
            statement = select(MemoryEmbeddingRecord).where(
                MemoryEmbeddingRecord.memory_id == memory_id
            )
            record = session.exec(statement).first()
            if record is None:
                record = MemoryEmbeddingRecord(
                    memory_id=memory_id,
                    project=project,
                    embedding_model=embedding_model,
                    vector_json=json.dumps(vector),
                )
            else:
                record.embedding_model = embedding_model
                record.vector_json = json.dumps(vector)
            session.add(record)
            session.commit()

    def list_for_project(self, project: str) -> list[MemoryEmbeddingRecord]:
        with Session(self.engine) as session:
            statement = select(MemoryEmbeddingRecord).where(
                MemoryEmbeddingRecord.project == project
            )
            return list(session.exec(statement).all())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMemoryService:
    """Embedding-based long-term memory with graceful text-search fallback."""

    def __init__(
        self,
        *,
        memory_repository: MemoryRepository,
        embedding_repository: MemoryEmbeddingRepository,
        embed: Callable[[str], list[float]] | None = None,
        query_embed: Callable[[str], list[float]] | None = None,
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        min_similarity: float = 0.35,
        top_k: int = 3,
    ) -> None:
        self.memory_repository = memory_repository
        self.embedding_repository = embedding_repository
        self.embedding_model = embedding_model
        self.min_similarity = min_similarity
        self.top_k = top_k
        client = OllamaEmbeddingClient(base_url=base_url, model=embedding_model)
        self._embed_document = embed or client.embed_document
        self._embed_query = query_embed or embed or client.embed_query

    def index(self, memory: PersonalMemoryRead) -> bool:
        """Embed and store a memory item. Returns False if embedding is unavailable."""
        try:
            vector = self._embed_document(f"{memory.title}\n{memory.content}")
        except Exception:
            return False
        self.embedding_repository.upsert(
            memory_id=memory.id,
            project=memory.project,
            embedding_model=self.embedding_model,
            vector=vector,
        )
        return True

    def search(
        self,
        *,
        project: str,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[PersonalMemoryRead, float]]:
        """Return (memory, similarity) pairs, best first.

        Falls back to SQLite text search (similarity reported as 0.0) when the
        embedding model is unavailable or nothing has been indexed.
        """
        limit = top_k or self.top_k
        records = self.embedding_repository.list_for_project(project)
        if records:
            try:
                query_vector = self._embed_query(query)
            except Exception:
                records = []
            else:
                scored: list[tuple[int, float]] = []
                for record in records:
                    try:
                        vector = json.loads(record.vector_json)
                    except json.JSONDecodeError:
                        continue
                    similarity = cosine_similarity(query_vector, vector)
                    if similarity >= self.min_similarity:
                        scored.append((record.memory_id, similarity))
                scored.sort(key=lambda pair: pair[1], reverse=True)
                results: list[tuple[PersonalMemoryRead, float]] = []
                memories = {
                    item.id: item
                    for item in self.memory_repository.search(project, "", limit=1000)
                }
                for memory_id, similarity in scored[:limit]:
                    memory = memories.get(memory_id)
                    if memory is not None:
                        results.append((memory, similarity))
                if results:
                    return results
        fallback = self.memory_repository.search(project, query, limit=limit)
        return [(memory, 0.0) for memory in fallback]

    def retrieve_facts(self, *, project: str, prompt: str, top_k: int | None = None) -> list[str]:
        """Format relevant memories for injection into backend context."""
        results = self.search(project=project, query=prompt, top_k=top_k)
        return [
            f"Long-term memory ({memory.title}): {memory.content}"
            for memory, similarity in results
            if similarity >= self.min_similarity
        ]
