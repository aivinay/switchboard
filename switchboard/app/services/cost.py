from __future__ import annotations

import math

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import CostEstimate, NormalizedRequest


class CostEstimator:
    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    def estimate_request_tokens(self, request: NormalizedRequest) -> int:
        text = "\n".join(message.content for message in request.messages)
        return self.estimate_text_tokens(text)

    def expected_output_tokens(self, request: NormalizedRequest) -> int:
        return request.max_tokens or 256

    def estimate(
        self,
        model: ModelProfile,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        input_cost = (input_tokens / 1_000_000) * model.input_cost_per_million_tokens
        output_cost = (output_tokens / 1_000_000) * model.output_cost_per_million_tokens
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(input_cost + output_cost, 8),
        )
