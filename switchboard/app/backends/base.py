from __future__ import annotations

from abc import ABC, abstractmethod

from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)


class AgentAdapter(ABC):
    name: str
    cost_type: BackendCostType

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def availability(self) -> BackendInfo:
        raise NotImplementedError

    @abstractmethod
    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        raise NotImplementedError
