from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine import Engine

from switchboard.app.core.config import Settings, resolve_config_file
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.personal import PersonalConfig
from switchboard.app.models.policy import PolicySet
from switchboard.app.providers.registry import ProviderRegistry
from switchboard.app.services.classifier import RequestClassifier
from switchboard.app.services.cost import CostEstimator
from switchboard.app.services.policy_engine import PolicyEngine
from switchboard.app.services.router import RoutingEngine
from switchboard.app.services.telemetry import TelemetryService
from switchboard.app.storage.repositories import (
    BackendMetricsRepository,
    ContextStore,
    MemoryRepository,
    PersonalTelemetryRepository,
    TelemetryRepository,
)


@dataclass
class ServiceContainer:
    settings: Settings
    catalogue: ModelCatalogue
    policies: PolicySet
    personal_config: PersonalConfig
    classifier: RequestClassifier
    cost_estimator: CostEstimator
    policy_engine: PolicyEngine
    router: RoutingEngine
    telemetry: TelemetryService
    personal_telemetry_repository: PersonalTelemetryRepository
    backend_metrics_repository: BackendMetricsRepository
    context_store: ContextStore
    memory_repository: MemoryRepository
    providers: ProviderRegistry


def build_container(settings: Settings, engine: Engine) -> ServiceContainer:
    catalogue = ModelCatalogue.from_yaml(settings.models_config_path)
    policies = PolicySet.from_yaml(settings.policies_config_path)
    personal_config = PersonalConfig.from_yaml(settings.personal_config_path)
    preferences = personal_config.preferences
    personal_config_dir = Path(settings.personal_config_path).expanduser().parent
    preferences.router_weights_path = resolve_config_file(
        preferences.router_weights_path,
        base_dir=personal_config_dir,
    )
    preferences.tool_dispatcher_weights_path = resolve_config_file(
        preferences.tool_dispatcher_weights_path,
        base_dir=personal_config_dir,
    )
    preferences.sensitivity_weights_path = resolve_config_file(
        preferences.sensitivity_weights_path,
        base_dir=personal_config_dir,
    )
    cost_estimator = CostEstimator()
    telemetry_repository = TelemetryRepository(engine)
    personal_telemetry_repository = PersonalTelemetryRepository(engine)
    backend_metrics_repository = BackendMetricsRepository(engine)
    context_store = ContextStore(engine)
    memory_repository = MemoryRepository(engine)
    ollama_base_url = personal_config.provider_base_url("ollama") or "http://localhost:11434"
    lmstudio_base_url = personal_config.provider_base_url("lmstudio") or "http://localhost:1234/v1"
    return ServiceContainer(
        settings=settings,
        catalogue=catalogue,
        policies=policies,
        personal_config=personal_config,
        classifier=RequestClassifier(),
        cost_estimator=cost_estimator,
        policy_engine=PolicyEngine(cost_estimator),
        router=RoutingEngine(catalogue, cost_estimator),
        telemetry=TelemetryService(telemetry_repository),
        personal_telemetry_repository=personal_telemetry_repository,
        backend_metrics_repository=backend_metrics_repository,
        context_store=context_store,
        memory_repository=memory_repository,
        providers=ProviderRegistry.default(cost_estimator, ollama_base_url, lmstudio_base_url),
    )
