#!/usr/bin/env python3
"""Small smoke comparison between BINT and a classical Beta Reputation System.

This is intentionally lightweight: it is not the thesis-scale evaluation notebook.
It runs the same delivery/map-sharing scenario for BINT (trust + staking) and BRS
(count-based beta reputation, no token staking) over a few seeds and prints a
compact metric table so that the BRS implementation can be sanity-checked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean
from typing import Any

from agents import DeliveryAgent, MaliciousDeliveryAgent
from model import AgentProfile, BintWorldModel, MAP_DATA_SERVICE


BASE_AGENT_KWARGS: dict[str, Any] = {
    "vision_radius": 2,
    "trust_reject_threshold": 0.30,
    "trust_accept_threshold": 0.80,
    "context_match_weight": 1.0,
    "other_context_weight": 0.25,
    "trust_prior_active": 1.0,
    "trust_prior_burned": 1.0,
    "burned_weight_multiplier": 1.0,
    "filter_untrusted_evidence": True,
    "max_negative_review_rate": 0.60,
    "min_reviews_before_reviewer_check": 8,
    "staking_min_fraction": 0.10,
    "staking_max_fraction": 0.90,
    "provider_stake_vtp_weight": 0.8,
    "provider_stake_reviewer_weight": 0.2,
    "requester_stake_vtp_weight": 0.4,
    "requester_stake_reviewer_weight": 0.6,
}

MALICIOUS_KWARGS: dict[str, Any] = {
    **BASE_AGENT_KWARGS,
    "false_map_probability": 0.50,
    "false_negative_review_probability": 0.50,
    "false_positive_review_probability": 0.50,
}


def build_profiles(honest_agents: int, malicious_agents: int) -> list[AgentProfile]:
    profiles: list[AgentProfile] = []
    if honest_agents > 0:
        profiles.append(AgentProfile(DeliveryAgent, honest_agents, dict(BASE_AGENT_KWARGS)))
    if malicious_agents > 0:
        profiles.append(
            AgentProfile(MaliciousDeliveryAgent, malicious_agents, dict(MALICIOUS_KWARGS))
        )
    return profiles


def build_model(trust_model: str, seed: int, args: argparse.Namespace) -> BintWorldModel:
    return BintWorldModel(
        rng=seed,
        width=args.width,
        height=args.height,
        num_drop_offs=args.dropoffs,
        genesis_tokens=args.genesis_tokens,
        max_steps=args.steps,
        staking_enabled=(trust_model == "bint"),
        trust_model=trust_model,
        brs_forgetting_factor=args.brs_forgetting_factor,
        agent_profiles=build_profiles(args.honest_agents, args.malicious_agents),
    )


def run_one(trust_model: str, seed: int, args: argparse.Namespace) -> dict[str, float | str | int]:
    model = build_model(trust_model, seed, args)
    for _ in range(args.steps):
        model.step()

    honest = [
        a for a in model.cached_delivery_agents
        if isinstance(a, DeliveryAgent) and not isinstance(a, MaliciousDeliveryAgent)
    ]
    malicious = [a for a in model.cached_delivery_agents if isinstance(a, MaliciousDeliveryAgent)]

    outcomes = len(model.outcomes)
    success_rate = model._n_success / outcomes if outcomes else 0.0
    failure_rate = model._n_failure / outcomes if outcomes else 0.0

    interactions = list(model.interactions.values())
    accepted_malicious_providers = 0
    for interaction in interactions:
        provider = model._find_delivery_agent(interaction.trustee_id)
        if isinstance(provider, MaliciousDeliveryAgent):
            accepted_malicious_providers += 1
    malicious_provider_rate = (
        accepted_malicious_providers / len(interactions) if interactions else 0.0
    )

    honest_scores = [
        a.calculate_trust_summary(a.unique_id, MAP_DATA_SERVICE)["score"] for a in honest
    ]
    malicious_scores = [
        a.calculate_trust_summary(a.unique_id, MAP_DATA_SERVICE)["score"] for a in malicious
    ]
    trust_gap = (
        (mean(honest_scores) if honest_scores else 0.0)
        - (mean(malicious_scores) if malicious_scores else 0.0)
    )

    return {
        "trust_model": trust_model,
        "seed": seed,
        "settled_interactions": outcomes,
        "total_interactions": len(interactions),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "malicious_provider_rate": malicious_provider_rate,
        "honest_points_per_agent": mean([a.points for a in honest]) if honest else 0.0,
        "malicious_points_per_agent": mean([a.points for a in malicious]) if malicious else 0.0,
        "trust_gap": trust_gap,
        "active_tnfts": sum(1 for t in model.tnft_ledger if t["status"]),
        "burned_tnfts": sum(1 for t in model.tnft_ledger if not t["status"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trust_model"]].append(row)

    metrics = [
        "settled_interactions",
        "success_rate",
        "failure_rate",
        "malicious_provider_rate",
        "honest_points_per_agent",
        "malicious_points_per_agent",
        "trust_gap",
        "active_tnfts",
        "burned_tnfts",
    ]
    result = []
    for trust_model, group in grouped.items():
        out: dict[str, Any] = {"trust_model": trust_model, "seeds": len(group)}
        for metric in metrics:
            out[metric] = mean(float(row[metric]) for row in group)
        result.append(out)
    return sorted(result, key=lambda row: row["trust_model"])


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("trust_model", "model"),
        ("seeds", "seeds"),
        ("settled_interactions", "settled"),
        ("success_rate", "success"),
        ("failure_rate", "failure"),
        ("malicious_provider_rate", "mal. provider"),
        ("honest_points_per_agent", "honest pts/agent"),
        ("malicious_points_per_agent", "mal. pts/agent"),
        ("trust_gap", "trust gap"),
        ("active_tnfts", "active TNFTs"),
        ("burned_tnfts", "burned TNFTs"),
    ]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    widths = [max(len(label), *(len(fmt(row[key])) for row in rows)) for key, label in columns]
    header = " | ".join(label.ljust(width) for (_, label), width in zip(columns, widths))
    sep = "-+-".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(fmt(row[key]).ljust(width) for (key, _), width in zip(columns, widths)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 1337, 2024])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--dropoffs", type=int, default=25)
    parser.add_argument("--honest-agents", type=int, default=14)
    parser.add_argument("--malicious-agents", type=int, default=6)
    parser.add_argument("--genesis-tokens", type=int, default=5)
    parser.add_argument("--brs-forgetting-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for trust_model in ("bint", "brs"):
        for seed in args.seeds:
            rows.append(run_one(trust_model, seed, args))
    print_table(aggregate(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
