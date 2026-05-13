from agents import DeliveryAgent, MaliciousDeliveryAgent
from model import AgentProfile


DEFAULT_WIDTH = 50
DEFAULT_HEIGHT = 50
DEFAULT_NUM_DROP_OFFS = 10
DEFAULT_GENESIS_TOKENS = 1
DEFAULT_RNG = 188422084722

DEFAULT_AGENT_VISION_RADIUS = 2
DEFAULT_TRUST_REJECT_THRESHOLD = 0.3
DEFAULT_TRUST_ACCEPT_THRESHOLD = 0.8

DEFAULT_HONEST_AGENTS = 7
DEFAULT_MALICIOUS_AGENTS = 3

DEFAULT_FALSE_MAP_PROBABILITY = 0.5
DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY = 0.5
DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY = 0.5
DEFAULT_MAX_NEGATIVE_REVIEW_RATE = 0.6
DEFAULT_MIN_REVIEWS_BEFORE_REVIEWER_CHECK = 3
DEFAULT_MAX_STEPS = 1_000

def default_agent_profiles() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_class=DeliveryAgent,
            count=DEFAULT_HONEST_AGENTS,
            kwargs={
                "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
                "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
                "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
                "max_negative_review_rate": DEFAULT_MAX_NEGATIVE_REVIEW_RATE,
                "min_reviews_before_reviewer_check": DEFAULT_MIN_REVIEWS_BEFORE_REVIEWER_CHECK,
            },
        ),
        AgentProfile(
            agent_class=MaliciousDeliveryAgent,
            count=DEFAULT_MALICIOUS_AGENTS,
            kwargs={
                "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
                "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
                "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
                "max_negative_review_rate": DEFAULT_MAX_NEGATIVE_REVIEW_RATE,
                "min_reviews_before_reviewer_check": DEFAULT_MIN_REVIEWS_BEFORE_REVIEWER_CHECK,
                "false_map_probability": DEFAULT_FALSE_MAP_PROBABILITY,
                "false_negative_review_probability": DEFAULT_FALSE_NEGATIVE_REVIEW_PROBABILITY,
                "false_positive_review_probability": DEFAULT_FALSE_POSITIVE_REVIEW_PROBABILITY,
            },
        ),
    ]

def honest_only_profiles() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_class=DeliveryAgent,
            count=DEFAULT_HONEST_AGENTS + DEFAULT_MALICIOUS_AGENTS,
            kwargs={
                "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
                "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
                "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
            },
        ),
    ]

def aggressive_malicious_profiles() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_class=DeliveryAgent,
            count=DEFAULT_HONEST_AGENTS,
            kwargs={
                "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
                "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
                "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
            },
        ),
        AgentProfile(
            agent_class=MaliciousDeliveryAgent,
            count=DEFAULT_MALICIOUS_AGENTS,
            kwargs={
                "vision_radius": DEFAULT_AGENT_VISION_RADIUS,
                "trust_reject_threshold": DEFAULT_TRUST_REJECT_THRESHOLD,
                "trust_accept_threshold": DEFAULT_TRUST_ACCEPT_THRESHOLD,
                "false_map_probability": 1.0,
                "false_negative_review_probability": 1.0,
                "false_positive_review_probability": 1.0,
            },
        ),
    ]

SCENARIOS = {
    "default": default_agent_profiles,
    "honest_only": honest_only_profiles,
    "aggressive_malicious": aggressive_malicious_profiles,
}


def get_agent_profiles(scenario_name: str) -> list[AgentProfile]:
    try:
        profile_factory = SCENARIOS[scenario_name]
    except KeyError:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid scenarios: {valid}")

    return profile_factory()
