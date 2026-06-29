from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from matplotlib.figure import Figure
from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter
import solara

from agents import DeliveryAgent, MaliciousDeliveryAgent
from model import BintWorldModel, MAP_DATA_SERVICE
from profiles import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_DROP_OFFS,
    DEFAULT_GENESIS_TOKENS,
    DEFAULT_RNG,
    DEFAULT_WIDTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_STAKING_ENABLED,
    DEFAULT_TRUST_REJECT_THRESHOLD,
    DEFAULT_TRUST_ACCEPT_THRESHOLD,
    SCENARIO_CONFIGS,
    get_agent_profiles,
    get_model_kwargs,
)

# -----------------------------------------------------------------------------
# Display constants
# -----------------------------------------------------------------------------

HONEST_AGENT_COLOR = "tab:blue"
MALICIOUS_AGENT_COLOR = "tab:red"
DROP_OFF_COLOR = "black"

HONEST_AGENT_MARKER = "o"
MALICIOUS_AGENT_MARKER = "X"
DROP_OFF_MARKER = "s"

APP_CSS = """
.bint-dashboard {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.bint-map {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.bint-lower-panel {
    width: 100%;
    display: grid;
    grid-template-columns: minmax(280px, 0.85fr) minmax(420px, 1.15fr);
    gap: 1rem;
    align-items: start;
}

.bint-summary-card,
.bint-figure {
    width: 100%;
    min-height: 260px;
}

.bint-figure {
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.bint-table-card {
    width: 100%;
    overflow-x: auto;
}

.bint-table-card table {
    width: 100%;
    min-width: 1200px;
    border-collapse: collapse;
    font-size: 0.82rem;
    table-layout: fixed;
}

.bint-table-card th,
.bint-table-card td {
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid #ddd;
    text-align: left;
    white-space: normal;
    overflow-wrap: anywhere;
    vertical-align: top;
}

.bint-table-card th {
    font-weight: 600;
}

.bint-map img,
.bint-map svg,
.bint-map canvas,
.bint-figure img,
.bint-figure svg,
.bint-figure canvas {
    max-width: 100% !important;
    max-height: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}

@media (max-width: 900px) {
    .bint-map {
        height: 56vh;
        min-height: 360px;
    }

    .bint-lower-panel {
        grid-template-columns: 1fr;
    }
}
"""

# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

DEFAULT_SCENARIO = "aggressive_malicious"

# Ordered list of scenario names for the UI selector, most illustrative first.
SCENARIO_OPTIONS = ["default", "brs_default", "honest_only", "aggressive_malicious"]


def make_bint_model(scenario_name: str = DEFAULT_SCENARIO) -> BintWorldModel:
    model_kwargs = {
        "staking_enabled": DEFAULT_STAKING_ENABLED,
        **get_model_kwargs(scenario_name),
    }
    model = BintWorldModel(
        rng=DEFAULT_RNG,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        num_drop_offs=DEFAULT_NUM_DROP_OFFS,
        genesis_tokens=DEFAULT_GENESIS_TOKENS,
        max_steps=DEFAULT_MAX_STEPS,
        agent_profiles=get_agent_profiles(scenario_name),
        **model_kwargs,
    )
    # Store the active scenario name so components can display it correctly.
    model.scenario_name = scenario_name
    return model


# -----------------------------------------------------------------------------
# Label dictionaries
# -----------------------------------------------------------------------------

DECISION_LABELS: dict[str, str] = {
    "accept_map_response": "accept response",
    "share_map": "share map",
    "none": "none",
}

REASON_LABELS: dict[str, str] = {
    # Acceptance
    "provider_accepted": "provider accepted",
    "requester_accepted": "requester accepted",
    # VTP / trust rejections
    "provider_rejected_low_vtp": "provider low VTP",
    "requester_rejected_low_vtp": "requester low VTP",
    # Reviewer credibility rejections
    "provider_rejected_low_reviewer_credibility": "provider reviewer risk",
    "requester_rejected_low_reviewer_credibility": "requester reviewer risk",
    # Stake rejections
    "provider_rejected_no_available_stake": "provider no stake",
    "requester_rejected_no_available_stake": "requester no stake",
    "provider_rejected_insufficient_stake": "provider stake too low",
    "requester_rejected_insufficient_stake": "requester stake too low",
    "provider_rejected_stake_limit": "provider stake limit",
    "requester_rejected_stake_limit": "requester stake limit",
    "stake_lock_failed": "stake lock failed",
    # Map sharing
    "rejected_unknown_target": "unknown target",
    "shared_false_map": "⚠ fabricated map",
    # Fallback
    "none": "none",
}

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _agent_groups(
    model: BintWorldModel,
) -> tuple[list[DeliveryAgent], list[MaliciousDeliveryAgent]]:
    malicious = [
        a for a in model.cached_delivery_agents if isinstance(a, MaliciousDeliveryAgent)
    ]
    honest = [
        a
        for a in model.cached_delivery_agents
        if isinstance(a, DeliveryAgent) and not isinstance(a, MaliciousDeliveryAgent)
    ]
    return honest, malicious


def _scatter_agents(
    ax: Any,
    agents: Iterable[DeliveryAgent],
    *,
    color: str,
    marker: str,
    size: float,
    label: str,
) -> None:
    agents = list(agents)
    if not agents:
        return
    xs = [a.cell.coordinate[0] for a in agents]
    ys = [a.cell.coordinate[1] for a in agents]
    ax.scatter(xs, ys, color=color, marker=marker, s=size, label=label, zorder=3)


def _trust_score(model: BintWorldModel, agent: DeliveryAgent) -> float:
    """Return agent's trust score from its own perspective."""
    return float(
        model.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE, evaluator=agent)[
            "score"
        ]
    )


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _short_id(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)[:12]


def _display_label(value: str | None, labels: dict[str, str]) -> str:
    if value is None:
        return "-"
    return labels.get(value, value.replace("_", " "))


def _model_thresholds(model: BintWorldModel) -> tuple[float, float]:
    """Read actual reject/accept thresholds from the first agent, with fallback."""
    agents = model.cached_delivery_agents
    if agents:
        a = agents[0]
        return a.trust_reject_threshold, a.trust_accept_threshold
    return DEFAULT_TRUST_REJECT_THRESHOLD, DEFAULT_TRUST_ACCEPT_THRESHOLD


# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------


@solara.component
def DynamicMap(model: BintWorldModel) -> None:
    """Render the current grid state with drop-offs and delivery agents."""
    update_counter.get()
    solara.Style(APP_CSS)

    grid_width = model.width
    grid_height = model.height

    dpi = 120
    target_cell_px = 12
    max_render_width_px = 720
    max_render_height_px = 620

    desired_width_px = max(180, grid_width * target_cell_px)
    desired_height_px = max(180, grid_height * target_cell_px)
    shrink = min(
        1.0,
        max_render_width_px / desired_width_px,
        max_render_height_px / desired_height_px,
    )

    render_width_px = max(180, int(desired_width_px * shrink))
    render_height_px = max(180, int(desired_height_px * shrink))
    effective_cell_px = min(
        render_width_px / max(grid_width, 1),
        render_height_px / max(grid_height, 1),
    )

    fig = Figure(
        figsize=(render_width_px / dpi, render_height_px / dpi),
        dpi=dpi,
        constrained_layout=False,
    )
    ax = fig.subplots()

    if effective_cell_px >= 4.0:
        line_width = max(0.03, min(0.15, effective_cell_px / 70))
        for x in range(grid_width + 1):
            ax.axvline(x - 0.5, lw=line_width, color="black", alpha=0.20, zorder=0)
        for y in range(grid_height + 1):
            ax.axhline(y - 0.5, lw=line_width, color="black", alpha=0.20, zorder=0)

    marker_area = max(10, min(55, (0.70 * effective_cell_px) ** 2))

    if model.cached_drop_offs:
        xs = [a.cell.coordinate[0] for a in model.cached_drop_offs]
        ys = [a.cell.coordinate[1] for a in model.cached_drop_offs]
        ax.scatter(xs, ys, color=DROP_OFF_COLOR, marker=DROP_OFF_MARKER,
                   s=marker_area, label="Drop-off", zorder=2)

    honest, malicious = _agent_groups(model)
    _scatter_agents(ax, honest, color=HONEST_AGENT_COLOR, marker=HONEST_AGENT_MARKER,
                    size=marker_area, label="Honest")
    _scatter_agents(ax, malicious, color=MALICIOUS_AGENT_COLOR, marker=MALICIOUS_AGENT_MARKER,
                    size=marker_area * 1.2, label="Malicious")

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if grid_width <= 80 and grid_height <= 80:
        ax.legend(loc="upper right", fontsize=7, frameon=True)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    return solara.Column(
        classes=["bint-map"],
        children=[solara.FigureMatplotlib(fig, format="png")],
    )


@solara.component
def CurrentTrustScores(model: BintWorldModel) -> None:
    """Horizontal bar chart of per-agent map-data trust scores."""
    update_counter.get()

    fig = Figure(figsize=(6.0, 3.8), dpi=120, constrained_layout=False)
    ax = fig.subplots()

    rows = [
        {
            "id": str(a.unique_id),
            "score": _trust_score(model, a),
            "is_malicious": isinstance(a, MaliciousDeliveryAgent),
        }
        for a in model.cached_delivery_agents
    ]

    if not rows:
        ax.set_title("Current Map-Data Trust Scores")
        ax.text(0.5, 0.5, "No delivery agents", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        rows.sort(key=lambda r: r["score"])
        labels = [r["id"] for r in rows]
        scores = [r["score"] for r in rows]
        colors = [
            MALICIOUS_AGENT_COLOR if r["is_malicious"] else HONEST_AGENT_COLOR
            for r in rows
        ]

        reject_thresh, accept_thresh = _model_thresholds(model)

        ax.barh(labels, scores, color=colors, alpha=0.75)
        ax.axvline(reject_thresh, color="black", linestyle=":", linewidth=1,
                   label=f"Reject ≤ {reject_thresh:.2f}")
        ax.axvline(accept_thresh, color="black", linestyle="--", linewidth=1,
                   label=f"Accept ≥ {accept_thresh:.2f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_title("Current Map-Data Trust Scores")
        ax.set_xlabel("Trust score")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout()
    return solara.Column(
        classes=["bint-figure"],
        children=[solara.FigureMatplotlib(fig, format="png")],
    )


@solara.component
def RunSummary(model: BintWorldModel, scenario_name: str) -> None:
    """Live summary card with mechanism-relevant metrics."""
    update_counter.get()

    honest, malicious = _agent_groups(model)

    # TNFT ledger counts
    active_tnfts = sum(1 for t in model.tnft_ledger if t["status"])
    burned_tnfts = sum(1 for t in model.tnft_ledger if not t["status"])
    staked_tnfts = sum(
        1 for t in model.tnft_ledger
        if t["status"] and t.get("staked_for") is not None
    )
    available_tnfts = active_tnfts - staked_tnfts

    # Delivery metrics
    total_deliveries = sum(a.delivery_count for a in model.cached_delivery_agents)
    total_points = sum(a.points for a in model.cached_delivery_agents)

    # Interaction safety metrics — the core BINT evaluation metrics
    n_outcomes = len(model.outcomes)
    success_rate = (
        model._n_success / n_outcomes if n_outcomes > 0 else None
    )
    failure_rate = (
        model._n_failure / n_outcomes if n_outcomes > 0 else None
    )

    # Trust gap: average honest score minus average malicious score
    honest_scores = [_trust_score(model, a) for a in honest]
    malicious_scores = [_trust_score(model, a) for a in malicious]
    avg_honest_trust = sum(honest_scores) / len(honest_scores) if honest_scores else None
    avg_malicious_trust = (
        sum(malicious_scores) / len(malicious_scores) if malicious_scores else None
    )
    trust_gap = (
        avg_honest_trust - avg_malicious_trust
        if avg_honest_trust is not None and avg_malicious_trust is not None
        else None
    )

    # Scenario description from config
    config = SCENARIO_CONFIGS.get(scenario_name)
    scenario_desc = config.description if config else scenario_name

    # Progress
    progress_pct = int(100 * model.steps / model.max_steps) if model.max_steps > 0 else 0

    with solara.Column(classes=["bint-summary-card"]):
        solara.Markdown(
            f"""
### Run summary

| | |
|---|---:|
| Scenario | {scenario_desc} |
| Trust model | {getattr(model, "trust_model", "bint").upper()} |
| Staking enabled | {"yes" if getattr(model, "staking_enabled", False) else "no"} |
| Progress | {model.steps} / {model.max_steps} steps ({progress_pct}%) |
| Honest agents | {len(honest)} |
| Malicious agents | {len(malicious)} |
| Drop-offs | {len(model.cached_drop_offs)} |

**Mechanism performance**

| Metric | Value |
|---|---:|
| Settled interactions | {n_outcomes} |
| Success rate | {_format_float(success_rate, 3)} |
| Failure rate | {_format_float(failure_rate, 3)} |
| Avg honest trust | {_format_float(avg_honest_trust, 3)} |
| Avg malicious trust | {_format_float(avg_malicious_trust, 3)} |
| Trust gap (H − M) | {_format_float(trust_gap, 3)} |

**Token economy**

| Metric | Value |
|---|---:|
| Active TNFTs | {active_tnfts} |
| Burned TNFTs | {burned_tnfts} |
| Staked TNFTs | {staked_tnfts} |
| Available TNFTs | {available_tnfts} |
| Total deliveries | {total_deliveries} |
| Total points | {total_points:.1f} |
"""
        )


@solara.component
def AgentDecisions(model: BintWorldModel) -> None:
    """Per-agent status table: trust, token economy, points, last decision."""
    update_counter.get()

    rows = []
    for agent in model.cached_delivery_agents:
        vtp_summary = model.get_vtp_summary(
            agent.unique_id, MAP_DATA_SERVICE, evaluator=agent
        )
        reviewer_summary = model.get_reviewer_summary(agent.unique_id)

        rows.append(
            {
                "id": _short_id(agent.unique_id),
                "type": "malicious" if isinstance(agent, MaliciousDeliveryAgent) else "honest",
                "trust": _format_float(float(vtp_summary["score"])),
                "reviews": reviewer_summary["total_reviews"],
                "negative": reviewer_summary["negative_reviews"],
                "neg_rate": _format_float(reviewer_summary["negative_review_rate"]),
                "active": vtp_summary["total_active"],
                "burned": vtp_summary["total_burned"],
                "points": f"{agent.points:.1f}",
                "deliveries": agent.delivery_count,
                "decision": _display_label(agent.last_decision_type, DECISION_LABELS),
                "reason": _display_label(agent.last_decision_reason, REASON_LABELS),
                "peer": _short_id(agent.last_decision_peer_id),
            }
        )

    table_rows = "\n".join(
        f"| {r['id']} | {r['type']} | {r['trust']} | "
        f"{r['reviews']} | {r['negative']} | {r['neg_rate']} | "
        f"{r['active']} | {r['burned']} | {r['points']} | {r['deliveries']} | "
        f"{r['decision']} | {r['reason']} | {r['peer']} |"
        for r in rows
    )

    with solara.Column(classes=["bint-table-card"]):
        solara.Markdown(
            f"""
### Agent status and decisions

| Agent | Type | Trust | Reviews | Neg. | Neg. rate | Active | Burned | Points | Deliveries | Last decision | Last reason | Last peer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
{table_rows}
"""
        )




@solara.component
def DashboardControls(model: BintWorldModel, reset_key: solara.Reactive[int]) -> None:
    """Small local controls for the custom dashboard page."""

    def refresh() -> None:
        update_counter.set(update_counter.get() + 1)

    def step_many(n: int) -> None:
        remaining = max(0, model.max_steps - model.steps)
        for _ in range(min(n, remaining)):
            model.step()
        refresh()

    with solara.Row(style="align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;"):
        solara.Button("Step 1", on_click=lambda: step_many(1))
        solara.Button("Step 50", on_click=lambda: step_many(50))
        solara.Button("Step 500", on_click=lambda: step_many(500))
        solara.Button("Reset scenario", on_click=lambda: reset_key.set(reset_key.value + 1))
        solara.Markdown(f"**Step:** {model.steps} / {model.max_steps}")


# -----------------------------------------------------------------------------
# Scenario selector + page
# -----------------------------------------------------------------------------

# Reactive state: tracks which scenario is currently displayed.
scenario_state: solara.Reactive[str] = solara.Reactive(DEFAULT_SCENARIO)


@solara.component
def BintPage() -> None:
    """Root component: scenario selector + full dashboard."""
    solara.Style(APP_CSS)

    # Re-create model when the selected scenario changes or when the reset button is pressed.
    scenario = scenario_state.value
    reset_key = solara.use_reactive(0)
    model = solara.use_memo(
        lambda: make_bint_model(scenario), dependencies=[scenario, reset_key.value]
    )

    with solara.Row(style="align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;"):
        solara.Select(
            label="Scenario",
            value=scenario,
            values=SCENARIO_OPTIONS,
            on_value=scenario_state.set,
            style="max-width: 360px;",
        )
        solara.Markdown(
            f"**Trust model:** {getattr(model, 'trust_model', 'bint').upper()} "
            f"| **Staking:** {'yes' if getattr(model, 'staking_enabled', False) else 'no'}"
        )

    DashboardControls(model, reset_key)

    with solara.Column(classes=["bint-dashboard"]):
        DynamicMap(model)
        with solara.Row(classes=["bint-lower-panel"]):
            RunSummary(model, scenario)
            CurrentTrustScores(model)
        AgentDecisions(model)


# -----------------------------------------------------------------------------
# Solara page entry point
# -----------------------------------------------------------------------------

@solara.component
def Page() -> None:
    BintPage()


# `solara run app.py` looks for `page`; point it to the custom page so the
# scenario selector is actually visible. The previous SolaraViz-only entry point
# worked for stepping, but it bypassed the selector component entirely.
page = Page
