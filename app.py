"""Solara dashboard for the BINT Mesa simulation.

This file is intentionally only the UI layer: it defines model controls, renders the
world map, and shows live analytics from the model's DataCollector. Simulation
rules and trust math should stay in `model.py` / `agents.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mesa.visualization import SolaraViz
import mesa.visualization.solara_viz as mesa_solara_viz
from mesa.visualization.utils import update_counter
import solara

from agents import (
    DEFAULT_TRUST_ACCEPT_THRESHOLD,
    DEFAULT_TRUST_REJECT_THRESHOLD,
    DeliveryAgent,
    DropOffLocationAgent,
    MaliciousMapDeliveryAgent,
)
from model import BintWorldModel, MAP_DATA_SERVICE

# Display constants
HONEST_AGENT_COLOR = "tab:blue"
MALICIOUS_AGENT_COLOR = "tab:red"
DROP_OFF_COLOR = "black"

HONEST_AGENT_MARKER = "o"
MALICIOUS_AGENT_MARKER = "X"
DROP_OFF_MARKER = "s"

DEFAULT_RNG = 188422084722
DEFAULT_WIDTH = 150
DEFAULT_HEIGHT = 150
DEFAULT_NUM_DELIVERY = 7
DEFAULT_NUM_MAP_MALICIOUS = 3
DEFAULT_NUM_DROP_OFFS = 15
DEFAULT_AGENT_VISION_RADIUS = 2
DEFAULT_GENESIS_TOKENS = 1
DEFAULT_MALICIOUSNESS_PROB = 0.5
DEFAULT_MAX_STEPS = 1_000


model_params = {
    "rng": {
        "type": "InputText",
        "value": DEFAULT_RNG,
        "label": "Random seed",
    },
    "width": {
        "type": "SliderInt",
        "value": DEFAULT_WIDTH,
        "label": "Grid width",
        "min": 5,
        "max": 300,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": DEFAULT_HEIGHT,
        "label": "Grid height",
        "min": 5,
        "max": 300,
        "step": 1,
    },
    "num_delivery": {
        "type": "SliderInt",
        "value": DEFAULT_NUM_DELIVERY,
        "label": "Honest delivery agents",
        "min": 0,
        "max": 30,
        "step": 1,
    },
    "num_map_malicious": {
        "type": "SliderInt",
        "value": DEFAULT_NUM_MAP_MALICIOUS,
        "label": "Malicious map agents",
        "min": 0,
        "max": 30,
        "step": 1,
    },
    "num_drop_offs": {
        "type": "SliderInt",
        "value": DEFAULT_NUM_DROP_OFFS,
        "label": "Drop-off locations",
        "min": 1,
        "max": 50,
        "step": 1,
    },
    "agent_vision_radius": {
        "type": "SliderInt",
        "value": DEFAULT_AGENT_VISION_RADIUS,
        "label": "Agent vision radius",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "genesis_tokens": {
        "type": "SliderInt",
        "value": DEFAULT_GENESIS_TOKENS,
        "label": "Bootstrap TNFTs per agent",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "maliciousness_prob": {
        "type": "SliderFloat",
        "value": DEFAULT_MALICIOUSNESS_PROB,
        "label": "Malicious lie probability",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "trust_reject_threshold": {
        "type": "SliderFloat",
        "value": DEFAULT_TRUST_REJECT_THRESHOLD,
        "label": "Trust auto-reject threshold",
        "min": 0.0,
        "max": 0.60,
        "step": 0.01,
    },
    "trust_accept_threshold": {
        "type": "SliderFloat",
        "value": DEFAULT_TRUST_ACCEPT_THRESHOLD,
        "label": "Trust auto-accept threshold",
        "min": 0.61,
        "max": 1.0,
        "step": 0.01,
    },
    "max_steps": {
        "type": "SliderInt",
        "value": DEFAULT_MAX_STEPS,
        "label": "Export after step",
        "min": 100,
        "max": 5_000,
        "step": 100,
    },
}


APP_CSS = """
.bint-map {
    width: 100%;
    height: 46vh;
    min-height: 280px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.bint-compact-figure {
    width: 100%;
    height: 25vh;
    min-height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.bint-map img,
.bint-map svg,
.bint-map canvas,
.bint-compact-figure img,
.bint-compact-figure svg,
.bint-compact-figure canvas {
    max-width: 100% !important;
    max-height: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}
"""


@solara.component
def AppStyles() -> None:
    solara.Style(APP_CSS)


def _patch_solara_viz_layout() -> None:
    """Make Mesa's default Solara grid layout more useful for this dashboard.

    This is deliberately defensive because `make_initial_grid_layout` is an
    internal Mesa visualization helper and may change between Mesa versions.
    """
    make_layout = getattr(mesa_solara_viz, "make_initial_grid_layout", None)
    if make_layout is None or getattr(make_layout, "_bint_patched", False):
        return

    def bigger_initial_layout(num_components: int) -> list[dict[str, Any]]:
        layout = make_layout(num_components)
        if not layout:
            return layout

        # The map should be the main visual element. Everything else stacks below.
        layout[0].update({"w": 12, "h": 7, "x": 0, "y": 0})

        for index, item in enumerate(layout[1:], start=1):
            item.update({"w": 12, "h": 3, "x": 0, "y": 7 + 3 * (index - 1)})

        return layout

    bigger_initial_layout._bint_patched = True  # type: ignore[attr-defined]
    mesa_solara_viz.make_initial_grid_layout = bigger_initial_layout


_patch_solara_viz_layout()


def _new_figure(
    width: float = 6.0, height: float = 3.0, dpi: int = 120
) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
    ax = fig.subplots()
    return fig, ax


def _agent_groups(
    model: BintWorldModel,
) -> tuple[list[DeliveryAgent], list[MaliciousMapDeliveryAgent]]:
    malicious = [
        agent
        for agent in model.cached_delivery_agents
        if isinstance(agent, MaliciousMapDeliveryAgent)
    ]
    honest = [
        agent
        for agent in model.cached_delivery_agents
        if isinstance(agent, DeliveryAgent)
        and not isinstance(agent, MaliciousMapDeliveryAgent)
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


@solara.component
def DynamicMap(model: BintWorldModel) -> None:
    """Render the current grid state with drop-offs, honest agents, and malicious agents."""
    update_counter.get()
    AppStyles()

    grid_width = model.width
    grid_height = model.height

    dpi = 120
    target_cell_px = 20
    max_render_width_px = 1_100
    max_render_height_px = 850

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

    # Full grid lines are useful for small grids but very noisy and slow for large ones.
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
        label="Malicious map agent",
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
def CompactFigureMatplotlib(fig: Figure) -> None:
    AppStyles()
    return solara.Column(
        classes=["bint-compact-figure"],
        children=[solara.FigureMatplotlib(fig, format="png")],
    )


def _empty_chart(ax: Any, title: str, message: str = "No data yet") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _get_agent_dataframe(model: BintWorldModel, agent_type: type[DeliveryAgent]):
    try:
        return model.datacollector.get_agenttype_vars_dataframe(agent_type)
    except (KeyError, RuntimeError):
        # RuntimeError can happen when Solara redraws while the model is stepping.
        return None


def _plot_agent_series(
    ax: Any,
    model: BintWorldModel,
    field_name: str,
    *,
    title: str,
    y_label: str,
    integer_y: bool = False,
) -> None:
    plotted_any = False

    for agent_type, color, linestyle, label_prefix in (
        (DeliveryAgent, HONEST_AGENT_COLOR, "-", "Honest"),
        (MaliciousMapDeliveryAgent, MALICIOUS_AGENT_COLOR, "--", "Malicious"),
    ):
        df = _get_agent_dataframe(model, agent_type)
        if df is None or df.empty or field_name not in df.columns:
            continue

        try:
            series_df = df.unstack(level="AgentID")[field_name]
        except (KeyError, ValueError):
            continue

        for index, agent_id in enumerate(series_df.columns):
            ax.plot(
                series_df.index,
                series_df[agent_id],
                color=color,
                linestyle=linestyle,
                linewidth=1.2 if agent_type is DeliveryAgent else 1.8,
                alpha=0.45 if agent_type is DeliveryAgent else 0.75,
                label=label_prefix if index == 0 else None,
            )
            plotted_any = True

    if not plotted_any:
        _empty_chart(ax, title)
        return

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.25)


@solara.component
def TNFTsOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()
    _plot_agent_series(
        ax,
        model,
        "active_tnfts",
        title="Active TNFTs by Agent",
        y_label="Active TNFTs",
        integer_y=True,
    )
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def BurnedTNFTsOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()
    _plot_agent_series(
        ax,
        model,
        "burned_tnfts",
        title="Burned TNFTs by Agent",
        y_label="Burned TNFTs",
        integer_y=True,
    )
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def PointsOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()
    _plot_agent_series(
        ax,
        model,
        "points",
        title="Agent Points Over Time",
        y_label="Points",
    )
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def DeliveriesOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()
    _plot_agent_series(
        ax,
        model,
        "delivery_count",
        title="Package Deliveries by Agent",
        y_label="Total Deliveries",
        integer_y=True,
    )
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def LedgerOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()

    try:
        df = model.datacollector.get_model_vars_dataframe()
    except RuntimeError:
        return CompactFigureMatplotlib(fig)

    if df.empty:
        _empty_chart(ax, "Ledger Over Time")
    else:
        if "active_ledger_size" in df.columns:
            ax.plot(
                df.index,
                df["active_ledger_size"],
                label="Active",
                color=HONEST_AGENT_COLOR,
            )
        if "burned_ledger_size" in df.columns:
            ax.plot(
                df.index,
                df["burned_ledger_size"],
                label="Burned",
                color=MALICIOUS_AGENT_COLOR,
            )

        ax.set_title("Global TNFT Ledger")
        ax.set_xlabel("Step")
        ax.set_ylabel("TNFTs")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7, frameon=False)

    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def OutcomesOverTime(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure()

    try:
        df = model.datacollector.get_model_vars_dataframe()
    except RuntimeError:
        return CompactFigureMatplotlib(fig)

    if df.empty:
        _empty_chart(ax, "Outcomes Over Time")
    else:
        if "success_outcomes" in df.columns:
            ax.plot(
                df.index,
                df["success_outcomes"],
                label="Successes",
                color=HONEST_AGENT_COLOR,
            )
        if "failure_outcomes" in df.columns:
            ax.plot(
                df.index,
                df["failure_outcomes"],
                label="Failures",
                color=MALICIOUS_AGENT_COLOR,
            )

        ax.set_title("Interaction Outcomes")
        ax.set_xlabel("Step")
        ax.set_ylabel("Outcomes")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7, frameon=False)

    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def CurrentTrustScores(model: BintWorldModel) -> None:
    update_counter.get()
    fig, ax = _new_figure(height=3.6)

    rows = []
    for agent in model.cached_delivery_agents:
        summary = model.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE)
        rows.append(
            {
                "id": str(agent.unique_id),
                "score": float(summary["score"]),
                "is_malicious": isinstance(agent, MaliciousMapDeliveryAgent),
            }
        )

    if not rows:
        _empty_chart(ax, "Current Map-Data Trust Scores", "No delivery agents")
        fig.tight_layout()
        return CompactFigureMatplotlib(fig)

    rows.sort(key=lambda row: row["score"])
    labels = [row["id"] for row in rows]
    scores = [row["score"] for row in rows]
    colors = [
        MALICIOUS_AGENT_COLOR if row["is_malicious"] else HONEST_AGENT_COLOR
        for row in rows
    ]

    ax.barh(labels, scores, color=colors, alpha=0.75)
    ax.axvline(
        model.trust_reject_threshold,
        color="black",
        linestyle=":",
        linewidth=1,
        label="Auto-reject",
    )
    ax.axvline(
        model.trust_accept_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Auto-accept",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Current Map-Data Trust Scores")
    ax.set_xlabel("Trust score")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def RunSummary(model: BintWorldModel) -> None:
    update_counter.get()

    honest, malicious = _agent_groups(model)
    trust_scores = [
        float(model.get_vtp_summary(agent.unique_id, MAP_DATA_SERVICE)["score"])
        for agent in model.cached_delivery_agents
    ]
    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

    successful_events = sum(
        1 for event in model.delivery_events if event.get("outcome_status") == "success"
    )
    failed_events = len(model.delivery_events) - successful_events

    solara.Markdown(f"""
### Run summary

| Metric | Value |
|---|---:|
| Step | {model.steps} |
| Honest agents | {len(honest)} |
| Malicious agents | {len(malicious)} |
| Drop-offs | {len(model.cached_drop_offs)} |
| Active TNFTs | {sum(1 for tnft in model.tnft_ledger if tnft["status"])} |
| Burned TNFTs | {sum(1 for tnft in model.tnft_ledger if not tnft["status"])} |
| Successful deliveries | {successful_events} |
| Failed deliveries | {failed_events} |
| Average map-data trust | {avg_trust:.3f} |
| Trust grey zone | {model.trust_reject_threshold:.2f} → {model.trust_accept_threshold:.2f} |
""")


@solara.component
def AnalyticsDashboard(model: BintWorldModel) -> None:
    with solara.lab.Tabs():
        with solara.lab.Tab("Summary"):
            RunSummary(model)
        with solara.lab.Tab("Trust Scores"):
            CurrentTrustScores(model)
        with solara.lab.Tab("Active TNFTs"):
            TNFTsOverTime(model)
        with solara.lab.Tab("Burned TNFTs"):
            BurnedTNFTsOverTime(model)
        with solara.lab.Tab("Ledger"):
            LedgerOverTime(model)
        with solara.lab.Tab("Outcomes"):
            OutcomesOverTime(model)
        with solara.lab.Tab("Deliveries"):
            DeliveriesOverTime(model)
        with solara.lab.Tab("Points"):
            PointsOverTime(model)


bint = BintWorldModel(
    rng=DEFAULT_RNG,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    num_delivery=DEFAULT_NUM_DELIVERY,
    num_map_malicious=DEFAULT_NUM_MAP_MALICIOUS,
    num_drop_offs=DEFAULT_NUM_DROP_OFFS,
    agent_vision_radius=DEFAULT_AGENT_VISION_RADIUS,
    trust_reject_threshold=DEFAULT_TRUST_REJECT_THRESHOLD,
    trust_accept_threshold=DEFAULT_TRUST_ACCEPT_THRESHOLD,
    genesis_tokens=DEFAULT_GENESIS_TOKENS,
    maliciousness_prob=DEFAULT_MALICIOUSNESS_PROB,
    max_steps=DEFAULT_MAX_STEPS,
)

page = SolaraViz(
    model=bint,
    components=[DynamicMap, (AnalyticsDashboard, 1)],
    model_params=model_params,
    name="BINT Simulation",
)
