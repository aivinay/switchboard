from __future__ import annotations

from switchboard.app.models.catalogue import ModelProfile, QualityTier
from switchboard.app.services.cost import CostEstimator


def test_token_estimation_uses_character_approximation() -> None:
    estimator = CostEstimator()

    assert estimator.estimate_text_tokens("abcd") == 1
    assert estimator.estimate_text_tokens("abcde") == 2


def test_cost_estimation_uses_model_prices() -> None:
    estimator = CostEstimator()
    model = ModelProfile(
        model_id="mock/test",
        provider="mock",
        display_name="Mock Test",
        context_window=1000,
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=3.0,
        average_latency_ms=100,
        supports_tools=False,
        supports_json_schema=True,
        supports_vision=False,
        allowed_regions=["uk"],
        data_policy="public",
        quality_tier=QualityTier.SMALL,
        enabled=True,
    )

    estimate = estimator.estimate(model, input_tokens=1_000, output_tokens=2_000)

    assert estimate.input_cost_usd == 0.001
    assert estimate.output_cost_usd == 0.006
    assert estimate.total_cost_usd == 0.007
