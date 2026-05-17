from __future__ import annotations
from profiles import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_DROP_OFFS,
    DEFAULT_GENESIS_TOKENS,
    DEFAULT_RNG,
    DEFAULT_WIDTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_TRUST_REJECT_THRESHOLD,
    DEFAULT_TRUST_ACCEPT_THRESHOLD,
    SCENARIOS,
    get_agent_profiles,
)

from collections.abc import Iterable
from typing import Any

from matplotlib.figure import Figure
from mesa.visualization import SolaraViz
import mesa.visualization.solara_viz as mesa_solara_viz
from mesa.visualization.utils import update_counter
import solara

from agents import DeliveryAgent, MaliciousDeliveryAgent
from model import BintWorldModel, MAP_DATA_SERVICE

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


@solara.component
def AppStyles() -> None:
    solara.Style(APP_CSS)


def _patch_solara_viz_layout() -> None:
    """Set better default component sizes for this dashboard.

    SolaraViz uses a resizable grid. Without this, wide/tall components like the
    decision table often start too small and need manual resizing.
    """
    make_layout = getattr(mesa_solara_viz, "make_initial_grid_layout", None)

    if make_layout is None or getattr(make_layout, "_bint_patched", False):
        return

    def bint_initial_layout(num_components: int) -> list[dict[str, Any]]:
        layout = make_layout(num_components)

        if not layout:
            return layout

        if num_components == 1:
            layout[0].update({"x": 0, "y": 0, "w": 12, "h": 10})
            return layout

        if num_components == 2:
            layout[0].update({"x": 0, "y": 0, "w": 8, "h": 8})
            layout[1].update({"x": 8, "y": 0, "w": 4, "h": 8})
            return layout

        for index, item in enumerate(layout):
            item.update({"x": 0, "y": index * 5, "w": 12, "h": 5})

        return layout

    bint_initial_layout._bint_patched = True  # type: ignore[attr-defined]
    mesa_solara_viz.make_initial_grid_layout = bint_initial_layout


_patch_solara_viz_layout()

# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

DEFAULT_SCENARIO = "aggressive_malicious"


def make_bint_model(scenario_name: str = DEFAULT_SCENARIO) -> BintWorldModel:
    return BintWorldModel(
        rng=DEFAULT_RNG,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        num_drop_offs=DEFAULT_NUM_DROP_OFFS,
        genesis_tokens=DEFAULT_GENESIS_TOKENS,
        max_steps=DEFAULT_MAX_STEPS,
        agent_profiles=get_agent_profiles(scenario_name),
    )


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _new_figure(
    width: float = 6.0,
    height: float = 3.0,
    dpi: int = 120,
) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
    ax = fig.subplots()
    return fig, ax


def _agent_groups(
    model: BintWorldModel,
) -> tuple[list[DeliveryAgent], list[MaliciousDeliveryAgent]]:
    malicious = [
        agent
        for agent in model.cached_delivery_agents
        if isinstance(agent, MaliciousDeliveryAgent)
    ]
    honest = [
        agent
        for agent in model.cached_delivery_agents
        if isinstance(agent, DeliveryAgent)
        and not isinstance(agent, MaliciousDeliveryAgent)
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

    xs = [agent.cell.coordinate[0] for agent in agents]
    ys = [agent.cell.coordinate[1] for agent in agents]
    ax.scatter(xs, ys, color=color, marker=marker, s=size, label=label, zorder=3)


def _trust_score(model: BintWorldModel, agent: DeliveryAgent) -> float:
    return float(model.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE)["score"])


def _format_optional_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"

    return f"{value:.{digits}f}"


def _short_id(value: Any) -> str:
    if value is None:
        return "-"

    text = str(value)
    return text[:8]


DECISION_LABELS = {
    "accept_map_response": "accept response",
    "share_map": "share map",
    "none": "none",
}

REASON_LABELS = {
    "provider_accepted": "provider accepted",
    "requester_accepted": "requester accepted",
    "provider_rejected_low_vtp": "provider low VTP",
    "provider_rejected_low_reviewer_credibility": "provider reviewer risk",
    "requester_rejected_low_vtp": "requester low VTP",
    "requester_rejected_low_reviewer_credibility": "requester reviewer risk",
    "rejected_unknown_target": "unknown target",
    "none": "none",
}


def _display_label(value: str | None, labels: dict[str, str]) -> str:
    if value is None:
        return "-"

    return labels.get(value, value.replace("_", " "))


# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------


@solara.component
def DynamicMap(model: BintWorldModel) -> None:
    """Render the current grid state with drop-offs and delivery agents."""

    update_counter.get()
    AppStyles()

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

    # Full grid lines are useful for small grids but noisy and slow for large ones.
    if effective_cell_px >= 4.0:
        line_width = max(0.03, min(0.15, effective_cell_px / 70))
        for x in range(grid_width + 1):
            ax.axvline(x - 0.5, lw=line_width, color="black", alpha=0.20, zorder=0)
        for y in range(grid_height + 1):
            ax.axhline(y - 0.5, lw=line_width, color="black", alpha=0.20, zorder=0)

    marker_area = max(10, min(55, (0.70 * effective_cell_px) ** 2))

    if model.cached_drop_offs:
        xs = [agent.cell.coordinate[0] for agent in model.cached_drop_offs]
        ys = [agent.cell.coordinate[1] for agent in model.cached_drop_offs]
        ax.scatter(
            xs,
            ys,
            color=DROP_OFF_COLOR,
            marker=DROP_OFF_MARKER,
            s=marker_area,
            label="Drop-off",
            zorder=2,
        )

    honest, malicious = _agent_groups(model)
    _scatter_agents(
        ax,
        honest,
        color=HONEST_AGENT_COLOR,
        marker=HONEST_AGENT_MARKER,
        size=marker_area,
        label="Honest delivery agent",
    )
    _scatter_agents(
        ax,
        malicious,
        color=MALICIOUS_AGENT_COLOR,
        marker=MALICIOUS_AGENT_MARKER,
        size=marker_area * 1.2,
        label="Malicious delivery agent",
    )

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
    """Show current map-data trust scores for all delivery agents."""

    update_counter.get()
    AppStyles()

    fig, ax = _new_figure(width=6.0, height=3.8)

    rows = []
    for agent in model.cached_delivery_agents:
        rows.append(
            {
                "id": str(agent.unique_id),
                "score": _trust_score(model, agent),
                "is_malicious": isinstance(agent, MaliciousDeliveryAgent),
            }
        )

    if not rows:
        ax.set_title("Current Map-Data Trust Scores")
        ax.text(
            0.5,
            0.5,
            "No delivery agents",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        rows.sort(key=lambda row: row["score"])
        labels = [row["id"] for row in rows]
        scores = [row["score"] for row in rows]
        colors = [
            MALICIOUS_AGENT_COLOR if row["is_malicious"] else HONEST_AGENT_COLOR
            for row in rows
        ]

        ax.barh(labels, scores, color=colors, alpha=0.75)
        ax.axvline(
            DEFAULT_TRUST_REJECT_THRESHOLD,
            color="black",
            linestyle=":",
            linewidth=1,
            label="Default auto-reject",
        )
        ax.axvline(
            DEFAULT_TRUST_ACCEPT_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Default auto-accept",
        )
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
    """Show a small live summary without relying on DataCollector."""

    update_counter.get()
    AppStyles()

    honest, malicious = _agent_groups(model)
    trust_scores = [
        _trust_score(model, agent) for agent in model.cached_delivery_agents
    ]
    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

    active_tnfts = sum(1 for tnft in model.tnft_ledger if tnft["status"])
    burned_tnfts = sum(1 for tnft in model.tnft_ledger if not tnft["status"])
    total_deliveries = sum(
        agent.delivery_count for agent in model.cached_delivery_agents
    )
    total_points = sum(agent.points for agent in model.cached_delivery_agents)

    with solara.Column(classes=["bint-summary-card"]):
        solara.Markdown(
            f"""
### Run summary

| Metric | Value |
|---|---:|
| Scenario | {scenario_name} |
| Step | {model.steps} |
| Honest agents | {len(honest)} |
| Malicious agents | {len(malicious)} |
| Drop-offs | {len(model.cached_drop_offs)} |
| Active TNFTs | {active_tnfts} |
| Burned TNFTs | {burned_tnfts} |
| Total deliveries | {total_deliveries} |
| Total points | {total_points:.2f} |
| Interactions | {len(model.interactions)} |
| Outcomes | {len(model.outcomes)} |
| Average map-data trust | {avg_trust:.3f} |
"""
        )


@solara.component
def RunSummaryPanel(model: BintWorldModel) -> None:
    RunSummary(model, DEFAULT_SCENARIO)


@solara.component
def AgentDecisions(model: BintWorldModel) -> None:
    """Show current per-agent status and latest recorded decision."""

    update_counter.get()
    AppStyles()

    rows = []
    for agent in model.cached_delivery_agents:
        vtp_summary = model.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE)
        reviewer_summary = model.get_reviewer_summary(agent.unique_id)

        rows.append(
            {
                "id": _short_id(agent.unique_id),
                "type": (
                    "malicious"
                    if isinstance(agent, MaliciousDeliveryAgent)
                    else "honest"
                ),
                "trust": _format_optional_float(float(vtp_summary["score"])),
                "reviews": reviewer_summary["total_reviews"],
                "negative": reviewer_summary["negative_reviews"],
                "neg_rate": _format_optional_float(
                    reviewer_summary["negative_review_rate"]
                ),
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
        f"| {row['id']} | {row['type']} | {row['trust']} | "
        f"{row['reviews']} | {row['negative']} | {row['neg_rate']} | "
        f"{row['active']} | {row['burned']} | {row['points']} | {row['deliveries']} | "
        f"{row['decision']} | {row['reason']} | {row['peer']} |"
        for row in rows
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
def Dashboard(model: BintWorldModel) -> None:
    """Single dashboard component to avoid Mesa/Solara grid overlap."""

    update_counter.get()
    AppStyles()

    with solara.Column(classes=["bint-dashboard"]):
        DynamicMap(model)
        with solara.Row(classes=["bint-lower-panel"]):
            RunSummary(model, DEFAULT_SCENARIO)
            CurrentTrustScores(model)
        AgentDecisions(model)


# -----------------------------------------------------------------------------
# Solara page
# -----------------------------------------------------------------------------


bint = make_bint_model()

page = SolaraViz(
    model=bint,
    components=[
        (DynamicMap, 0),
        (RunSummaryPanel, 0),
        (AgentDecisions, 1),
        (CurrentTrustScores, 2),
    ],
    name="BINT Simulation",
)
