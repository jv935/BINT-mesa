"""
Agent profile definitions and scenario configurations for the BINT dashboard.

A "scenario" is a named configuration that specifies how many honest and
malicious agents to spawn and with what parameters.  Three scenarios are
provided for the interactive dashboard:

    default             3 honest + 2 malicious, medium attack probability (0.5).
                        Mirrors the main ablation setting from the thesis.

    honest_only         5 honest agents, no malicious agents.
                        Upper-bound reference: what does the system look like
                        when every agent behaves correctly?

    aggressive_malicious  3 honest + 2 malicious, all attack probabilities = 1.0.
                          Worst-case adversarial setting for the dashboard.

Note: the evaluation notebook uses larger populations (14 honest / 6 malicious
on a 150x150 grid).  The defaults below are sized for the interactive dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import DeliveryAgent, MaliciousDeliveryAgent
from model import AgentProfile


# ---------------------------------------------------------------------------
# World / model defaults (used by both dashboard and model factory)
# ---------------------------------------------------------------------------

DEFAULT_WIDTH = 50
DEFAULT_HEIGHT = 50
DEFAULT_NUM_DROP_OFFS = 5
DEFAULT_GENESIS_TOKENS = 5
DEFAULT_RNG = None
DEFAULT_MAX_STEPS = 2_000
DEFAULT_STAKING_ENABLED = True
DEFAULT_TRUST_MODEL = "bint"

# ---------------------------------------------------------------------------
# Agent behaviour defaults
# ---------------------------------------------------------------------------

DEFAULT_AGENT_VISION_RADIUS = 2

DEFAULT_TRUST_REJECT_THRESHOLD = 0.3
DEFAULT_TRUST_ACCEPT_THRESHOLD = 0.8

DEFAULT_HONEST_AGENTS = 3
DEFAULT_MALICIOUS_AGENTS = 2

DEFAULT_FALSE_MAP_PROBABILITY = 0.5
DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY = 0.5
DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY = 0.5

DEFAULT_MAX_NEGATIVE_REVIEW_RATE = 0.6

# Keep consistent with the experiments in the thesis (Section 6.4 uses 8).
# The DeliveryAgent __init__ default is 10; the dashboard uses this value.
DEFAULT_MIN_REVIEWS_BEFORE_REVIEWER_CHECK = 8

DEFAULT_STAKING_MIN_FRACTION = 0.25
DEFAULT_STAKING_MAX_FRACTION = 1.0

DEFAULT_PROVIDER_STAKE_VTP_WEIGHT = 0.8
DEFAULT_PROVIDER_STAKE_REVIEWER_WEIGHT = 0.2
DEFAULT_REQUESTER_STAKE_VTP_WEIGHT = 0.4
DEFAULT_REQUESTER_STAKE_REVIEWER_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# Shared kwargs builders
# ---------------------------------------------------------------------------

def _base_agent_kwargs() -> dict[str, Any]:
    """Return the kwargs common to every delivery agent (honest or malicious)."""
    return {
        "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
        "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
        "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
        "max_negative_review_rate": DEFAULT_MAX_NEGATIVE_REVIEW_RATE,
        "min_reviews_before_reviewer_check": DEFAULT_MIN_REVIEWS_BEFORE_REVIEWER_CHECK,
        "staking_min_fraction": DEFAULT_STAKING_MIN_FRACTION,
        "staking_max_fraction": DEFAULT_STAKING_MAX_FRACTION,
        "provider_stake_vtp_weight": DEFAULT_PROVIDER_STAKE_VTP_WEIGHT,
        "provider_stake_reviewer_weight": DEFAULT_PROVIDER_STAKE_REVIEWER_WEIGHT,
        "requester_stake_vtp_weight": DEFAULT_REQUESTER_STAKE_VTP_WEIGHT,
        "requester_stake_reviewer_weight": DEFAULT_REQUESTER_STAKE_REVIEWER_WEIGHT,
    }


def _malicious_agent_kwargs(
    false_map_probability: float = DEFAULT_FALSE_MAP_PROBABILITY,
    false_negative_review_probability: float = DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY,
    false_positive_review_probability: float = DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY,
) -> dict[str, Any]:
    """Return kwargs for a malicious agent, layered on top of the base kwargs."""
    return {
        **_base_agent_kwargs(),
        "false_map_probability": false_map_probability,
        "false_negative_review_probability": false_negative_review_probability,
        "false_positive_review_probability": false_positive_review_probability,
    }


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Declarative description of one simulation scenario.

    Attributes
    ----------
    name:
        Machine-readable identifier used as the key in SCENARIOS.
    description:
        One-line human-readable summary shown in the dashboard selector.
    num_honest:
        Number of honest DeliveryAgents to spawn.
    num_malicious:
        Number of MaliciousDeliveryAgents to spawn.
    false_map_probability:
        Probability that a malicious provider fabricates a coordinate.
    false_negative_review_probability:
        Probability that a malicious requester falsely reports failure.
    false_positive_review_probability:
        Probability that a malicious requester falsely reports success.
    trust_model:
        Trust mechanism used by the model: ``"bint"`` or ``"brs"``.
    staking_enabled:
        Whether BINT staking is enabled. This is ignored by the model for BRS.
    """

    name: str
    description: str
    num_honest: int
    num_malicious: int
    false_map_probability: float = DEFAULT_FALSE_MAP_PROBABILITY
    false_negative_review_probability: float = DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY
    false_positive_review_probability: float = DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY
    trust_model: str = DEFAULT_TRUST_MODEL
    staking_enabled: bool = DEFAULT_STAKING_ENABLED

    def build_profiles(self) -> list[AgentProfile]:
        """Construct the AgentProfile list for this scenario."""
        profiles: list[AgentProfile] = []

        if self.num_honest > 0:
            profiles.append(
                AgentProfile(
                    agent_class=DeliveryAgent,
                    count=self.num_honest,
                    kwargs=_base_agent_kwargs(),
                )
            )

        if self.num_malicious > 0:
            profiles.append(
                AgentProfile(
                    agent_class=MaliciousDeliveryAgent,
                    count=self.num_malicious,
                    kwargs=_malicious_agent_kwargs(
                        false_map_probability=self.false_map_probability,
                        false_negative_review_probability=self.false_negative_review_probability,
                        false_positive_review_probability=self.false_positive_review_probability,
                    ),
                )
            )

        if not profiles:
            raise ValueError(
                f"Scenario '{self.name}' has no agents "
                f"(num_honest={self.num_honest}, num_malicious={self.num_malicious})."
            )

        return profiles


# ---------------------------------------------------------------------------
# Built-in scenario definitions
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict[str, ScenarioConfig] = {
    "default": ScenarioConfig(
        name="default",
        description="3 honest + 2 malicious (medium attacks, p=0.5)",
        num_honest=DEFAULT_HONEST_AGENTS,
        num_malicious=DEFAULT_MALICIOUS_AGENTS,
        false_map_probability=DEFAULT_FALSE_MAP_PROBABILITY,
        false_negative_review_probability=DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY,
        false_positive_review_probability=DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY,
    ),
    "honest_only": ScenarioConfig(
        name="honest_only",
        description="5 honest agents only — upper-bound reference",
        num_honest=DEFAULT_HONEST_AGENTS + DEFAULT_MALICIOUS_AGENTS,
        num_malicious=0,
    ),
    "aggressive_malicious": ScenarioConfig(
        name="aggressive_malicious",
        description="3 honest + 2 malicious (maximum attacks, p=1.0)",
        num_honest=DEFAULT_HONEST_AGENTS,
        num_malicious=DEFAULT_MALICIOUS_AGENTS,
        false_map_probability=1.0,
        false_negative_review_probability=1.0,
        false_positive_review_probability=1.0,
    ),
    "brs_default": ScenarioConfig(
        name="brs_default",
        description="BRS baseline: 3 honest + 2 malicious (medium attacks, p=0.5)",
        num_honest=DEFAULT_HONEST_AGENTS,
        num_malicious=DEFAULT_MALICIOUS_AGENTS,
        false_map_probability=DEFAULT_FALSE_MAP_PROBABILITY,
        false_negative_review_probability=DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY,
        false_positive_review_probability=DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY,
        trust_model="brs",
        staking_enabled=False,
    ),
}

# Keep the old SCENARIOS dict as a convenience alias so that existing callers
# (eval notebook, parameter exploration notebook) continue to work unchanged.
SCENARIOS: dict[str, ScenarioConfig] = SCENARIO_CONFIGS


def get_scenario_config(scenario_name: str) -> ScenarioConfig:
    """Return the named ScenarioConfig, or raise a helpful ValueError."""
    config = SCENARIO_CONFIGS.get(scenario_name)
    if config is None:
        valid = ", ".join(sorted(SCENARIO_CONFIGS))
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Valid scenarios: {valid}"
        )
    return config


def get_model_kwargs(scenario_name: str) -> dict[str, Any]:
    """Return model-level kwargs associated with a scenario."""
    config = get_scenario_config(scenario_name)
    return {
        "trust_model": config.trust_model,
        "staking_enabled": config.staking_enabled,
    }


def get_agent_profiles(scenario_name: str) -> list[AgentProfile]:
    """Return the AgentProfile list for the named scenario.

    Raises ValueError for unknown scenario names.
    """
    return get_scenario_config(scenario_name).build_profiles()
