import argparse
import csv
import os
import sys
from typing import Any

import numpy as np
from mesa import batch_run

from model import BintWorldModel


CSV_FILE = "bint_batch_results.csv"
EXPORT_DIR = "exports"

DEFAULT_MAX_STEPS = 1500
DEFAULT_SIZE = (150, 150)
DEFAULT_AGENT_VISION_RADIUS = 2

DROP_OFF_COUNTS = [5, 10, 15]
TOTAL_DELIVERY_AGENTS = 10
MALICIOUS_COUNTS = range(TOTAL_DELIVERY_AGENTS + 1)

# representative setting used for ablations and most figures
ABLATION_DROP_OFFS = 10
ABLATION_TOTAL_DELIVERY_AGENTS = 10
ABLATION_MALICIOUS = 3


def make_scenario(
        *,
        scenario_family: str,
        scenario_name: str,
        num_drop_offs: int,
        total_delivery_agents: int,
        num_map_malicious: int,
        trust_threshold: float = 0.5,
        genesis_tokens: int = 3,
        maliciousness_prob: float = 0.5,
        max_steps: int = DEFAULT_MAX_STEPS,
        export_dir: str = EXPORT_DIR,
) -> dict[str, Any]:
    if num_map_malicious > total_delivery_agents:
        raise ValueError("num_map_malicious cannot exceed total_delivery_agents.")

    num_delivery = total_delivery_agents - num_map_malicious

    return {
        "scenario_family": scenario_family,
        "scenario_name": scenario_name,
        "size": DEFAULT_SIZE,
        "num_drop_offs": num_drop_offs,
        "agent_vision_radius": DEFAULT_AGENT_VISION_RADIUS,
        "num_delivery": num_delivery,
        "num_map_malicious": num_map_malicious,
        "trust_threshold": trust_threshold,
        "genesis_tokens": genesis_tokens,
        "maliciousness_prob": maliciousness_prob,
        "max_steps": max_steps,
        "export_dir": export_dir,
    }


def build_robustness_scenarios(max_steps: int, export_dir: str) -> list[dict[str, Any]]:
    scenarios = []

    for num_drop_offs in DROP_OFF_COUNTS:
        for num_map_malicious in MALICIOUS_COUNTS:
            scenarios.append(
                make_scenario(
                    scenario_family="robustness",
                    scenario_name=f"robustness_drops_{num_drop_offs}_mal_{num_map_malicious}",
                    num_drop_offs=num_drop_offs,
                    total_delivery_agents=TOTAL_DELIVERY_AGENTS,
                    num_map_malicious=num_map_malicious,
                    max_steps=max_steps,
                    export_dir=export_dir,
                )
            )

    return scenarios


def build_ablation_scenarios(max_steps: int, export_dir: str) -> list[dict[str, Any]]:
    base_kwargs = {
        "num_drop_offs": ABLATION_DROP_OFFS,
        "total_delivery_agents": ABLATION_TOTAL_DELIVERY_AGENTS,
        "num_map_malicious": ABLATION_MALICIOUS,
        "max_steps": max_steps,
        "export_dir": export_dir,
    }

    return [
        make_scenario(
            scenario_family="ablation",
            scenario_name="current_bint",
            trust_threshold=0.5,
            genesis_tokens=3,
            maliciousness_prob=0.5,
            **base_kwargs,
        ),
        make_scenario(
            scenario_family="ablation",
            scenario_name="no_trust_gate",
            trust_threshold=0.0,
            genesis_tokens=3,
            maliciousness_prob=0.5,
            **base_kwargs,
        ),
        make_scenario(
            scenario_family="ablation",
            scenario_name="strict_bint",
            trust_threshold=1.0,
            genesis_tokens=3,
            maliciousness_prob=0.5,
            **base_kwargs,
        ),
        make_scenario(
            scenario_family="ablation",
            scenario_name="no_bootstrap_trust",
            trust_threshold=0.5,
            genesis_tokens=0,
            maliciousness_prob=0.5,
            **base_kwargs,
        ),
        make_scenario(
            scenario_family="ablation",
            scenario_name="all_lies_when_malicious",
            trust_threshold=0.5,
            genesis_tokens=3,
            maliciousness_prob=1.0,
            **base_kwargs,
        ),
        make_scenario(
            scenario_family="ablation",
            scenario_name="no_malicious_agents",
            num_drop_offs=ABLATION_DROP_OFFS,
            total_delivery_agents=ABLATION_TOTAL_DELIVERY_AGENTS,
            num_map_malicious=0,
            trust_threshold=0.5,
            genesis_tokens=3,
            maliciousness_prob=0.0,
            max_steps=max_steps,
            export_dir=export_dir,
        ),
    ]


def build_smoke_scenarios(max_steps: int, export_dir: str) -> list[dict[str, Any]]:
    return [
        make_scenario(
            scenario_family="smoke",
            scenario_name="smoke_current_bint",
            num_drop_offs=5,
            total_delivery_agents=5,
            num_map_malicious=2,
            trust_threshold=0.5,
            genesis_tokens=3,
            maliciousness_prob=0.5,
            max_steps=max_steps,
            export_dir=export_dir,
        )
    ]


def build_scenarios(suite: str, max_steps: int, export_dir: str) -> list[dict[str, Any]]:
    if suite == "smoke":
        return build_smoke_scenarios(max_steps, export_dir)
    if suite == "robustness":
        return build_robustness_scenarios(max_steps, export_dir)
    if suite == "ablation":
        return build_ablation_scenarios(max_steps, export_dir)
    if suite == "main":
        return build_robustness_scenarios(max_steps, export_dir) + build_ablation_scenarios(max_steps, export_dir)

    raise ValueError(f"Unknown suite: {suite}")


def append_rows(csv_filename: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())
    write_header = not os.path.exists(csv_filename)

    with open(csv_filename, "a", newline="", buffering=10_485_760) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BINT evaluation sweeps.")

    parser.add_argument("chunk_size", nargs="?", type=int, default=25)
    parser.add_argument("csv_filename", nargs="?", default=CSV_FILE)

    parser.add_argument("--suite", choices=["main", "robustness", "ablation", "smoke"], default="main")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rng_values = rng.integers(0, sys.maxsize, size=args.chunk_size).tolist()
    scenarios = build_scenarios(
        suite=args.suite,
        max_steps=args.max_steps,
        export_dir=args.export_dir,
    )

    print(
        f"Running suite={args.suite} | scenarios={len(scenarios)} | "
        f"seed replications per scenario in this chunk={args.chunk_size}"
    )

    total_rows = 0

    for scenario_index, scenario in enumerate(scenarios, start=1):
        parameters = {key: [value] for key, value in scenario.items()}

        print(
            f"[{scenario_index}/{len(scenarios)}] "
            f"{scenario['scenario_name']} "
            f"(honest={scenario['num_delivery']}, malicious={scenario['num_map_malicious']})"
        )

        results = batch_run(
            model_cls=BintWorldModel,
            parameters=parameters,
            rng=rng_values,
            max_steps=args.max_steps,
            number_processes=args.processes,
            data_collection_period=1,
            display_progress=True,
        )

        append_rows(args.csv_filename, results)
        total_rows += len(results)

    print(f"Success! Wrote {total_rows} rows to {args.csv_filename}")
