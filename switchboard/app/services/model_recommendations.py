from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from switchboard.app.models.catalogue import ModelCatalogue, ModelKind, ModelProfile

GIB = 1024**3

CHAT_ROLE_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "floor": {
        "general": ["ollama/llama3.2:3b"],
        "coding": ["ollama/llama3.2:3b"],
        "reasoning": ["ollama/llama3.2:3b"],
    },
    "16gb": {
        "general": ["ollama/gemma4:e4b", "ollama/llama3.2:3b"],
        "coding": ["ollama/qwen3.5:9b"],
        "reasoning": ["ollama/gpt-oss:20b"],
    },
    "32gb": {
        "general": ["ollama/gemma4:12b", "ollama/gemma4:e4b"],
        "coding": ["ollama/qwen3.5:9b"],
        "reasoning": ["ollama/gpt-oss:20b", "ollama/glm-4.7-flash"],
    },
    "48gb": {
        "general": ["ollama/gemma4:31b", "ollama/gemma4:26b", "ollama/gemma4:12b"],
        "coding": ["ollama/qwen3-coder:30b", "ollama/qwen3.6:27b", "ollama/qwen3.5:9b"],
        "reasoning": ["ollama/glm-4.7-flash", "ollama/gpt-oss:20b"],
    },
}

EMBEDDING_CANDIDATES = [
    "ollama/embeddinggemma",
    "ollama/qwen3-embedding:0.6b",
    "ollama/nomic-embed-text",
]

FLOOR_EMBEDDING_CANDIDATES = ["ollama/embeddinggemma"]

FLOOR_TIER_NOTE = (
    "Detected RAM is below the local model pack floor. Switchboard recommends "
    "llama3.2:3b for every chat role; heavier local models need more RAM, so "
    "quota-aware routing to premium backends matters more on small machines."
)

APPLY_ROLE_MAP = {
    "simple": "general",
    "private_summary": "general",
    "general": "general",
    "coding": "coding",
    "reasoning": "reasoning",
    "complex_coding": "coding",
}


@dataclass(frozen=True)
class ModelRecommendation:
    role: str
    model_id: str
    ollama_tag: str
    display_name: str
    notes: str | None = None


@dataclass(frozen=True)
class LocalModelPackRecommendation:
    total_ram_bytes: int | None
    tier: str
    roles: list[ModelRecommendation]
    notes: list[str]

    @property
    def pull_commands(self) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        for role in self.roles:
            if role.ollama_tag in seen:
                continue
            seen.add(role.ollama_tag)
            commands.append(f"ollama pull {role.ollama_tag}")
        return commands


def parse_linux_meminfo_total_bytes(text: str) -> int | None:
    for line in text.splitlines():
        if not line.startswith("MemTotal:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]) * 1024
    return None


def parse_sysctl_memsize_bytes(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if stripped.isdigit() else None


def detect_total_ram_bytes() -> int | None:
    system = platform.system()
    if system == "Linux":
        meminfo = Path("/proc/meminfo")
        try:
            return parse_linux_meminfo_total_bytes(meminfo.read_text(encoding="utf-8"))
        except OSError:
            return None
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return parse_sysctl_memsize_bytes(result.stdout)
    return None


def ram_tier(total_ram_bytes: int | None) -> str:
    if total_ram_bytes is None:
        return "16gb"
    if total_ram_bytes < 12 * GIB:
        return "floor"
    if total_ram_bytes <= 18 * GIB:
        return "16gb"
    if total_ram_bytes < 48 * GIB:
        return "32gb"
    return "48gb"


def recommend_local_model_pack(
    catalogue: ModelCatalogue,
    *,
    total_ram_bytes: int | None = None,
) -> LocalModelPackRecommendation:
    tier = ram_tier(total_ram_bytes)
    by_id = {model.model_id: model for model in catalogue.models}
    roles: list[ModelRecommendation] = []
    for role, candidates in CHAT_ROLE_CANDIDATES[tier].items():
        model = _first_selectable_chat_model(by_id, candidates)
        if model is not None:
            roles.append(_recommendation(role, model))
    embedding_candidates = (
        FLOOR_EMBEDDING_CANDIDATES if tier == "floor" else EMBEDDING_CANDIDATES
    )
    embedding = _first_enabled_embedding_model(by_id, embedding_candidates)
    if embedding is not None:
        roles.append(_recommendation("embeddings", embedding))
    return LocalModelPackRecommendation(
        total_ram_bytes=total_ram_bytes,
        tier=tier,
        roles=roles,
        notes=[FLOOR_TIER_NOTE] if tier == "floor" else [],
    )


def apply_local_model_pack(
    *,
    personal_config_path: str | Path,
    models_config_path: str | Path,
    recommendation: LocalModelPackRecommendation,
) -> None:
    personal_path = Path(personal_config_path)
    models_path = Path(models_config_path)
    role_by_name = {role.role: role for role in recommendation.roles}

    personal_payload = _load_yaml_mapping(personal_path)
    preferences = personal_payload.setdefault("preferences", {})
    if not isinstance(preferences, dict):
        preferences = {}
        personal_payload["preferences"] = preferences
    local_model_roles = preferences.setdefault("local_model_roles", {})
    if not isinstance(local_model_roles, dict):
        local_model_roles = {}
        preferences["local_model_roles"] = local_model_roles
    for config_role, recommendation_role in APPLY_ROLE_MAP.items():
        selected = role_by_name.get(recommendation_role)
        if selected is not None:
            local_model_roles[config_role] = selected.model_id
    embedding = role_by_name.get("embeddings")
    if embedding is not None:
        preferences["embedding_model"] = embedding.ollama_tag
    _write_yaml_mapping(personal_path, personal_payload)

    models_payload = _load_yaml_mapping(models_path)
    selected_ids = {role.model_id for role in recommendation.roles}
    models_value = models_payload.get("models", [])
    if isinstance(models_value, list):
        for model in models_value:
            if isinstance(model, dict) and model.get("model_id") in selected_ids:
                model["enabled"] = True
    _write_yaml_mapping(models_path, models_payload)


def _first_selectable_chat_model(
    by_id: dict[str, ModelProfile],
    candidates: list[str],
) -> ModelProfile | None:
    for model_id in candidates:
        model = by_id.get(model_id)
        if (
            model is not None
            and model.enabled
            and model.provider == "ollama"
            and model.is_chat_selectable
        ):
            return model
    return None


def _first_enabled_embedding_model(
    by_id: dict[str, ModelProfile],
    candidates: list[str],
) -> ModelProfile | None:
    for model_id in candidates:
        model = by_id.get(model_id)
        if model is not None and model.enabled and model.kind == ModelKind.LOCAL_EMBEDDING:
            return model
    return None


def _recommendation(role: str, model: ModelProfile) -> ModelRecommendation:
    return ModelRecommendation(
        role=role,
        model_id=model.model_id,
        ollama_tag=model.provider_model_name or model.model_id.split("/", 1)[-1],
        display_name=model.display_name,
        notes=model.notes,
    )


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _write_yaml_mapping(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
