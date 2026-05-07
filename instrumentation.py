"""Reporting and export helpers for the BINT simulation.

This module keeps observation/reporting code out of ``model.py`` so the model
can focus on simulation state and behaviour.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import uuid

import mesa
from mesa.discrete_space import CellAgent

from agents import DeliveryAgent, MaliciousMapDeliveryAgent


def get_agent_type(agent: CellAgent) -> str:
    return type(agent).__name__


def get_ledger_size(model: mesa.Model) -> int:
    return len(model.tnft_ledger)


def get_active_ledger_size(model: mesa.Model) -> int:
    return sum(1 for tnft in model.tnft_ledger if tnft["status"])


def get_burned_ledger_size(model: mesa.Model) -> int:
    return sum(1 for tnft in model.tnft_ledger if not tnft["status"])


def get_total_points(model: mesa.Model) -> float:
    return float(sum(agent.points for agent in model.cached_delivery_agents))


def get_total_deliveries(model: mesa.Model) -> int:
    return sum(agent.delivery_count for agent in model.cached_delivery_agents)


def get_pending_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "pending")


def get_completed_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "completed")


def get_cancelled_interactions(model: mesa.Model) -> int:
    return sum(1 for record in model.interactions.values() if record.status == "cancelled")


def get_success_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "success")


def get_failure_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "failure")


def get_disputed_outcomes(model: mesa.Model) -> int:
    return sum(1 for record in model.outcomes.values() if record.status == "disputed")


def get_delivery_events(model: mesa.Model) -> int:
    return len(model.delivery_events)


def get_successful_delivery_events(model: mesa.Model) -> int:
    return sum(1 for event in model.delivery_events if event["outcome_status"] == "success")


def get_failed_delivery_events(model: mesa.Model) -> int:
    return sum(1 for event in model.delivery_events if event["outcome_status"] != "success")


def get_agent_map_size(agent: CellAgent) -> int:
    return getattr(agent, "map_size", 0)


def get_agent_current_provider_id(agent: CellAgent) -> str:
    return getattr(agent, "current_provider_id", None) or ""


def get_agent_goal_name(agent: CellAgent) -> str:
    return getattr(agent, "goal_name", None) or ""


def get_agent_cell_x(agent: CellAgent) -> int | None:
    return agent.cell.coordinate[0] if getattr(agent, "cell", None) is not None else None


def get_agent_cell_y(agent: CellAgent) -> int | None:
    return agent.cell.coordinate[1] if getattr(agent, "cell", None) is not None else None


def get_agent_target_x(agent: CellAgent) -> int | None:
    target = getattr(agent, "target_coordinate", None)
    return target[0] if target is not None else None


def get_agent_target_y(agent: CellAgent) -> int | None:
    target = getattr(agent, "target_coordinate", None)
    return target[1] if target is not None else None


def get_agent_package_min_steps(agent: CellAgent) -> int | None:
    package = getattr(agent, "package", None)
    return package["min_steps"] if package is not None else None


def get_agent_package_max_steps(agent: CellAgent) -> int | None:
    package = getattr(agent, "package", None)
    return package["max_steps"] if package is not None else None


def build_data_collector() -> mesa.DataCollector:
    """Create the Mesa data collector used during a simulation run."""

    tracking_parameters = {
        "agent_type": get_agent_type,
        "state": "state",
        "points": "points",
        "delivery_count": "delivery_count",
        "active_tnfts": "cached_active_tnfts",
        "burned_tnfts": "cached_burned_tnfts",
        "known_drop_offs": "known_drop_offs_count",
        "map_size": get_agent_map_size,
        "steps_on_package": "steps_on_package",
        "package_min_steps": get_agent_package_min_steps,
        "package_max_steps": get_agent_package_max_steps,
        "goal_name": get_agent_goal_name,
        "current_provider_id": get_agent_current_provider_id,
        "cell_x": get_agent_cell_x,
        "cell_y": get_agent_cell_y,
        "target_x": get_agent_target_x,
        "target_y": get_agent_target_y,
    }

    return mesa.DataCollector(
        model_reporters={
            "ledger_size": get_ledger_size,
            "active_ledger_size": get_active_ledger_size,
            "burned_ledger_size": get_burned_ledger_size,
            "total_points": get_total_points,
            "total_deliveries": get_total_deliveries,
            "pending_interactions": get_pending_interactions,
            "completed_interactions": get_completed_interactions,
            "cancelled_interactions": get_cancelled_interactions,
            "success_outcomes": get_success_outcomes,
            "failure_outcomes": get_failure_outcomes,
            "disputed_outcomes": get_disputed_outcomes,
            "delivery_events": get_delivery_events,
            "successful_delivery_events": get_successful_delivery_events,
            "failed_delivery_events": get_failed_delivery_events,
        },
        agenttype_reporters={
            DeliveryAgent: tracking_parameters,
            MaliciousMapDeliveryAgent: tracking_parameters,
        },
    )


def build_agent_snapshot(agent: DeliveryAgent) -> dict[str, Any]:
    """Return the end-of-run summary for one delivery agent."""

    return {
        "agent_id": agent.unique_id,
        "agent_type": type(agent).__name__,
        "is_malicious_type": isinstance(agent, MaliciousMapDeliveryAgent),
        "maliciousness": getattr(agent, "maliciousness", 0.0),
        "cell": agent.cell.coordinate,
        "final_state": agent.state,
        "final_points": agent.points,
        "total_deliveries": agent.delivery_count,
        "active_tnfts": agent.cached_active_tnfts,
        "burned_tnfts": agent.cached_burned_tnfts,
        "known_drop_offs": agent.known_drop_offs_count,
        "map_size": agent.map_size,
    }


def safe_name(value: str) -> str:
    """Convert a scenario name into a filesystem-safe filename component."""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def build_export_payload(model: mesa.Model, run_id: str) -> dict[str, Any]:
    """Build the JSON-serialisable end-of-run payload."""

    return {
        "run_id": run_id,
        "rng_seed": model.rng_seed,
        "scenario": {
            "scenario_name": model.scenario_name,
            "scenario_family": model.scenario_family,
        },
        "config": {
            "size": model.size,
            "width": model.width,
            "height": model.height,
            "num_drop_offs": model.num_drop_offs,
            "num_delivery": model.num_delivery,
            "num_map_malicious": model.num_map_malicious,
            "total_delivery_agents": model.total_delivery_agents,
            "agent_vision_radius": model.agent_vision_radius,
            "trust_reject_threshold": model.trust_reject_threshold,
            "trust_accept_threshold": model.trust_accept_threshold,
            "genesis_tokens": model.genesis_tokens,
            "maliciousness_prob": model.maliciousness_prob,
            "max_steps": model.max_steps,
        },
        "total_steps": model.steps,
        "summary": {
            "ledger_size": get_ledger_size(model),
            "active_ledger_size": get_active_ledger_size(model),
            "burned_ledger_size": get_burned_ledger_size(model),
            "total_points": get_total_points(model),
            "total_deliveries": get_total_deliveries(model),
            "pending_interactions": get_pending_interactions(model),
            "completed_interactions": get_completed_interactions(model),
            "cancelled_interactions": get_cancelled_interactions(model),
            "success_outcomes": get_success_outcomes(model),
            "failure_outcomes": get_failure_outcomes(model),
            "delivery_events": get_delivery_events(model),
            "successful_delivery_events": get_successful_delivery_events(model),
            "failed_delivery_events": get_failed_delivery_events(model),
        },
        "drop_offs": [
            {
                "id": drop_off.unique_id,
                "coord": drop_off.cell.coordinate,
                "coord_x": drop_off.cell.coordinate[0],
                "coord_y": drop_off.cell.coordinate[1],
            }
            for drop_off in model.cached_drop_offs
        ],
        "agent_snapshots": [
            build_agent_snapshot(agent) for agent in model.cached_delivery_agents
        ],
        "interactions": [asdict(record) for record in model.interactions.values()],
        "outcomes": [asdict(record) for record in model.outcomes.values()],
        "delivery_events": model.delivery_events,
        "tnft_ledger": model.tnft_ledger,
    }


def export_end_of_run_data(model: mesa.Model) -> Path:
    """Write the end-of-run JSON export and return the output path."""

    export_dir = Path(model.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    filename = export_dir / f"{safe_name(model.scenario_name)}_seed_{model.rng_seed}_{run_id}.json"

    with filename.open("w") as f:
        json.dump(build_export_payload(model, run_id), f, indent=4)

    return filename
