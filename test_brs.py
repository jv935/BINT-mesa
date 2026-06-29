from __future__ import annotations

from agents import DeliveryAgent
from model import AgentProfile, BintWorldModel, MAP_DATA_SERVICE


def _two_agent_brs_model() -> BintWorldModel:
    return BintWorldModel(
        width=10,
        height=10,
        num_drop_offs=2,
        agent_profiles=[AgentProfile(DeliveryAgent, 2)],
        genesis_tokens=5,
        staking_enabled=True,
        trust_model="brs",
        rng=123,
    )


def test_brs_mode_disables_token_staking_and_uses_neutral_prior() -> None:
    model = _two_agent_brs_model()
    evaluator, target = model.cached_delivery_agents[:2]

    assert model.trust_model == "brs"
    assert model.staking_enabled is False
    assert model.tnft_ledger == []

    summary = evaluator.calculate_trust_summary(target.unique_id, MAP_DATA_SERVICE)
    assert summary["trust_calculator"] == "BetaReputationSystem"
    assert summary["score"] == 0.5
    assert summary["successes"] == 0.0
    assert summary["failures"] == 0.0


def test_brs_updates_from_settled_success_and_failure() -> None:
    model = _two_agent_brs_model()
    evaluator, target = model.cached_delivery_agents[:2]

    interaction_id = model.record_interaction(
        truster_id=evaluator.unique_id,
        trustee_id=target.unique_id,
        service_type=MAP_DATA_SERVICE,
    )
    model.settle_interaction(interaction_id, evaluator.unique_id, "success")

    summary_after_success = evaluator.calculate_trust_summary(
        target.unique_id, MAP_DATA_SERVICE
    )
    assert summary_after_success["successes"] == 1.0
    assert summary_after_success["failures"] == 0.0
    assert summary_after_success["score"] == 2.0 / 3.0
    assert model.tnft_ledger == []

    interaction_id = model.record_interaction(
        truster_id=evaluator.unique_id,
        trustee_id=target.unique_id,
        service_type=MAP_DATA_SERVICE,
    )
    model.settle_interaction(interaction_id, evaluator.unique_id, "failure")

    summary_after_failure = evaluator.calculate_trust_summary(
        target.unique_id, MAP_DATA_SERVICE
    )
    assert summary_after_failure["successes"] == 1.0
    assert summary_after_failure["failures"] == 1.0
    assert summary_after_failure["score"] == 0.5
    assert model.tnft_ledger == []


def test_brs_forgetting_factor_decays_old_evidence() -> None:
    model = BintWorldModel(
        width=10,
        height=10,
        num_drop_offs=2,
        agent_profiles=[AgentProfile(DeliveryAgent, 2)],
        trust_model="brs",
        brs_forgetting_factor=0.5,
        rng=123,
    )
    evaluator, target = model.cached_delivery_agents[:2]

    for status in ("success", "success"):
        interaction_id = model.record_interaction(
            truster_id=evaluator.unique_id,
            trustee_id=target.unique_id,
            service_type=MAP_DATA_SERVICE,
        )
        model.settle_interaction(interaction_id, evaluator.unique_id, status)

    evidence = model.get_brs_evidence(target.unique_id, MAP_DATA_SERVICE)
    # First success is decayed once before the second one is added: 0.5 + 1.0.
    assert evidence["successes"] == 1.5
    assert evidence["failures"] == 0.0
