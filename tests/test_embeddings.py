from __future__ import annotations

import json

import httpx

from switchboard.app.models.personal import PersonalMemoryRead
from switchboard.app.models.telemetry import MemoryEmbeddingRecord
from switchboard.app.services.semantic_memory import (
    NOMIC_EMBED_CONTEXT,
    OllamaEmbeddingClient,
    SemanticMemoryService,
)


class FakeResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"embedding": self.embedding}


def test_nomic_embeddings_use_task_prefixes_and_num_ctx(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse([1.0, 2.0])

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaEmbeddingClient(model="nomic-embed-text")

    assert client.embed_classification("route me") == [1.0, 2.0]
    assert client.embed_document("saved fact") == [1.0, 2.0]
    assert client.embed_query("find fact") == [1.0, 2.0]

    payloads = [call["json"] for call in calls]
    assert payloads[0]["prompt"] == "classification: route me"
    assert payloads[1]["prompt"] == "search_document: saved fact"
    assert payloads[2]["prompt"] == "search_query: find fact"
    assert payloads[0]["options"] == {"num_ctx": NOMIC_EMBED_CONTEXT}


def test_qwen_embedding_uses_instruction_prompt_without_nomic_options(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> FakeResponse:
        calls.append(json)
        return FakeResponse([3.0])

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaEmbeddingClient(model="qwen3-embedding:0.6b")

    assert client.embed_query("router docs") == [3.0]

    assert calls[0]["model"] == "qwen3-embedding:0.6b"
    assert "retrieving relevant saved memories" in str(calls[0]["prompt"])
    assert "options" not in calls[0]


def test_semantic_memory_uses_document_and_query_embedders() -> None:
    class FakeMemoryRepository:
        def __init__(self) -> None:
            self.memory = PersonalMemoryRead(
                id=1,
                project="personal",
                title="Preference",
                content="Prefer local models.",
                tags=[],
                created_at="now",
            )

        def search(self, project: str, query: str, limit: int) -> list[PersonalMemoryRead]:
            return [self.memory]

    class FakeEmbeddingRepository:
        def __init__(self) -> None:
            self.records: list[MemoryEmbeddingRecord] = []

        def upsert(
            self,
            *,
            memory_id: int,
            project: str,
            embedding_model: str,
            vector: list[float],
        ) -> None:
            self.records = [
                MemoryEmbeddingRecord(
                    memory_id=memory_id,
                    project=project,
                    embedding_model=embedding_model,
                    vector_json=json.dumps(vector),
                )
            ]

        def list_for_project(self, project: str) -> list[MemoryEmbeddingRecord]:
            return self.records

    document_inputs: list[str] = []
    query_inputs: list[str] = []
    memory_repository = FakeMemoryRepository()
    embedding_repository = FakeEmbeddingRepository()
    service = SemanticMemoryService(
        memory_repository=memory_repository,  # type: ignore[arg-type]
        embedding_repository=embedding_repository,  # type: ignore[arg-type]
        embed=lambda text: document_inputs.append(text) or [1.0],
        query_embed=lambda text: query_inputs.append(text) or [1.0],
        min_similarity=0.0,
    )

    assert service.index(memory_repository.memory)
    assert service.search(project="personal", query="local preference")
    assert document_inputs == ["Preference\nPrefer local models."]
    assert query_inputs == ["local preference"]
