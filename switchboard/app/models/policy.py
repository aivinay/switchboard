from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import Sensitivity


class TenantPolicy(BaseModel):
    policy_id: str
    tenant_id: str
    workflow_id: str = "default"
    version: str
    allowed_providers: list[str] = Field(default_factory=list)
    blocked_providers: list[str] = Field(default_factory=list)
    allowed_models: list[str] = Field(default_factory=list)
    blocked_models: list[str] = Field(default_factory=list)
    max_cost_per_request_usd: float | None = None
    max_latency_ms: int | None = None
    allowed_sensitivity_levels: list[Sensitivity] = Field(default_factory=list)
    require_private_model_for_regulated_data: bool = False
    allow_prompt_logging: bool = False
    allow_response_logging: bool = False
    fallback_model: str | None = None
    default_routing_mode: str = "observe"


class PolicySet(BaseModel):
    policies: list[TenantPolicy]

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySet:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(policies=[TenantPolicy(**item) for item in payload.get("policies", [])])

    def match(self, tenant_id: str, workflow_id: str) -> TenantPolicy:
        exact = [
            policy
            for policy in self.policies
            if policy.tenant_id == tenant_id and policy.workflow_id == workflow_id
        ]
        if exact:
            return exact[0]

        tenant_default = [
            policy
            for policy in self.policies
            if policy.tenant_id == tenant_id and policy.workflow_id == "default"
        ]
        if tenant_default:
            return tenant_default[0]

        default = [
            policy
            for policy in self.policies
            if policy.tenant_id == "default" and policy.workflow_id == "default"
        ]
        if not default:
            raise ValueError("policy configuration must include default/default policy")
        return default[0]


class PolicyDecision(BaseModel):
    allowed: bool
    reason_codes: list[str] = Field(default_factory=list)
    candidate_models: list[ModelProfile] = Field(default_factory=list)
    policy_version: str
    policy_id: str
