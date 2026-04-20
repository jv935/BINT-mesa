import mesa
import pandas as pd
from matplotlib.ticker import MaxNLocator
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
import mesa.visualization.solara_viz as mesa_solara_viz
import solara
from solara import lab
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

from agents import DeliveryAgent, DropOffLocationAgent, MaliciousMapDeliveryAgent
from model import BintWorldModel

rng = 188422084722
width = 150
height = 150
num_delivery = 7
num_map_malicious = 3
num_drop_offs = 15

model_params = {
    "rng": {
        "type": "InputText",
        "value": rng,
        "label": "Random seed",
    },
    "width": {
        "type": "SliderInt",
        "value": width,
        "label": "Width",
        "min": 1,
        "max": 300,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": height,
        "label": "Height",
        "min": 1,
        "max": 300,
        "step": 1,
    },
    "num_delivery": {
        "type": "SliderInt",
        "value": num_delivery,
        "label": "Number of honest delivery agents",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "num_map_malicious": {
        "type": "SliderInt",
        "value": num_map_malicious,
        "label": "Number of malicious agents",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "num_drop_offs": {
        "type": "SliderInt",
        "value": num_drop_offs,
        "label": "Number of drop offs",
        "min": 1,
        "max": 30,
        "step": 1,
    },
}


# def agent_portrayal(agent: mesa.Agent):
#     if isinstance(agent, MaliciousMapDeliveryAgent):
#         return AgentPortrayalStyle(size=20, color="red")
#     elif isinstance(agent, DeliveryAgent):
#         return AgentPortrayalStyle(size=20, color="blue")
#     elif isinstance(agent, DropOffLocationAgent):
#         return AgentPortrayalStyle(size=20, marker="s", color="black")
#     else:
#         return None
#
#
# def property_layer_portrayal(layer: mesa.discrete_space.PropertyLayer):
#     if layer.name == "drop_off_locations":
#         return PropertyLayerStyle(color="black", alpha=0.8)
#
#     return None
#
#
# bint = BintWorldModel()
#
# renderer = SpaceRenderer(model=bint, backend="matplotlib")
# renderer.draw_structure(lw=0.2, ls="solid", color="black", alpha=0.5)
# renderer.setup_agents(agent_portrayal).draw_agents()


def _patch_solara_viz_layout() -> None:
    """Make page components start larger instead of using Mesa's default 6x10 tiles."""
    if getattr(mesa_solara_viz.make_initial_grid_layout, "_bint_patched", False):
        return

    original_make_initial_grid_layout = mesa_solara_viz.make_initial_grid_layout

    def bigger_initial_layout(num_components: int):
        layout = original_make_initial_grid_layout(num_components)
        if num_components == 1:
            layout[0]["w"] = 12
            layout[0]["h"] = 11
            layout[0]["x"] = 0
            layout[0]["y"] = 0
            return layout

        if num_components >= 2:
            layout[0]["w"] = 12
            layout[0]["h"] = 6
            layout[0]["x"] = 0
            layout[0]["y"] = 0

            layout[1]["w"] = 12
            layout[1]["h"] = 2
            layout[1]["x"] = 0
            layout[1]["y"] = 6

            for i, item in enumerate(layout[2:], start=2):
                item["w"] = 12
                item["h"] = 2
                item["x"] = 0
                item["y"] = 6 + 2 * (i - 1)
            return layout

        return layout

    bigger_initial_layout._bint_patched = True
    mesa_solara_viz.make_initial_grid_layout = bigger_initial_layout


_patch_solara_viz_layout()


@solara.component
def DynamicMap(model: BintWorldModel):
    update_counter.get()

    solara.Style(
        """
        .dynamic-map {
            width: 100%;
            height: 40vh;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .dynamic-map img,
        .dynamic-map svg,
        .dynamic-map canvas {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }
        """
    )

    grid_width = model.grid.width
    grid_height = model.grid.height

    dpi = 120
    target_cell_px = 20
    max_render_width_px = 1000
    max_render_height_px = 800

    desired_width_px = max(160, grid_width * target_cell_px)
    desired_height_px = max(160, grid_height * target_cell_px)

    shrink = min(
        1.0,
        max_render_width_px / desired_width_px,
        max_render_height_px / desired_height_px,
    )

    render_width_px = max(160, int(desired_width_px * shrink))
    render_height_px = max(160, int(desired_height_px * shrink))
    effective_cell_px = min(render_width_px / max(grid_width, 1), render_height_px / max(grid_height, 1))

    fig = Figure(
        figsize=(render_width_px / dpi, render_height_px / dpi),
        dpi=dpi,
        constrained_layout=False,
    )
    ax = fig.subplots()

    line_width = max(0.03, min(0.15, effective_cell_px / 70))
    marker_area = max(6, min(28, (0.55 * effective_cell_px) ** 2))

    for x in range(grid_width + 1):
        ax.axvline(x - 0.5, lw=line_width, color="black", alpha=0.25)
    for y in range(grid_height + 1):
        ax.axhline(y - 0.5, lw=line_width, color="black", alpha=0.25)

    if model.cached_drop_offs:
        xs = [a.cell.coordinate[0] for a in model.cached_drop_offs]
        ys = [a.cell.coordinate[1] for a in model.cached_drop_offs]
        ax.scatter(xs, ys, color="black", marker="s", s=marker_area)

    honest = [a for a in model.cached_delivery_agents if type(a) is DeliveryAgent]
    if honest:
        xs = [a.cell.coordinate[0] for a in honest]
        ys = [a.cell.coordinate[1] for a in honest]
        ax.scatter(xs, ys, color="blue", marker="o", s=marker_area)

    malicious = [a for a in model.cached_delivery_agents if type(a) is MaliciousMapDeliveryAgent]
    if malicious:
        xs = [a.cell.coordinate[0] for a in malicious]
        ys = [a.cell.coordinate[1] for a in malicious]
        ax.scatter(xs, ys, color="red", marker="o", s=marker_area)

    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    return solara.Column(
        classes=["dynamic-map"],
        children=[solara.FigureMatplotlib(fig, format="png")],
    )


@solara.component
def CompactFigureMatplotlib(fig):
    solara.Style(
        """
        .compact-figure {
            width: 100%;
            height: 20vh;
            min-height: 170px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .compact-figure img,
        .compact-figure svg,
        .compact-figure canvas {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }
        """
    )

    return solara.Column(
        classes=["compact-figure"],
        children=[solara.FigureMatplotlib(fig, format="png")],
    )


@solara.component
def TNFTsOverTime(model: BintWorldModel):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()

    try:
        df_honest = model.datacollector.get_agenttype_vars_dataframe(DeliveryAgent)
        df_malicious = model.datacollector.get_agenttype_vars_dataframe(MaliciousMapDeliveryAgent)
    except RuntimeError:
        # catch the multithreading collision and skip this frame
        return CompactFigureMatplotlib(fig)

    if not df_honest.empty:
        vtp_df = df_honest.unstack(level="AgentID")["Active TNFTs"]
        for agent_id in vtp_df.columns:
            ax.plot(vtp_df.index, vtp_df[agent_id], alpha=0.7)

    if not df_malicious.empty:
        vtp_df = df_malicious.unstack(level="AgentID")["Active TNFTs"]
        for agent_id in vtp_df.columns:
            ax.plot(vtp_df.index, vtp_df[agent_id], linestyle="--", linewidth=2)

    ax.set_title("Verifiable Trust Portfolio (Active TNFTs)")
    ax.set_xlabel("Step")
    ax.set_ylabel("TNFT Balance")

    ax.axhline(y=1, color="gray", linestyle=":", label="Trust Threshold")
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    return CompactFigureMatplotlib(fig)


@solara.component
def PointsOverTime(model: BintWorldModel):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()

    try:
        df_honest = model.datacollector.get_agenttype_vars_dataframe(DeliveryAgent)
        df_malicious = model.datacollector.get_agenttype_vars_dataframe(MaliciousMapDeliveryAgent)
    except RuntimeError:
        return CompactFigureMatplotlib(fig)

    if not df_honest.empty:
        points_df = df_honest.unstack(level="AgentID")["Points"]
        for agent_id in points_df.columns:
            ax.plot(points_df.index, points_df[agent_id], alpha=0.7)

    if not df_malicious.empty:
        points_df = df_malicious.unstack(level="AgentID")["Points"]
        for agent_id in points_df.columns:
            ax.plot(points_df.index, points_df[agent_id], linestyle="--", linewidth=2)

    ax.set_title("Agent Points Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Points")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def DeliveriesOverTime(model: BintWorldModel):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()

    try:
        df_honest = model.datacollector.get_agenttype_vars_dataframe(DeliveryAgent)
        df_malicious = model.datacollector.get_agenttype_vars_dataframe(MaliciousMapDeliveryAgent)
    except RuntimeError:
        return CompactFigureMatplotlib(fig)

    if not df_honest.empty:
        del_df = df_honest.unstack(level="AgentID")["Deliveries"]
        for agent_id in del_df.columns:
            ax.plot(del_df.index, del_df[agent_id], alpha=0.7)

    if not df_malicious.empty:
        del_df = df_malicious.unstack(level="AgentID")["Deliveries"]
        for agent_id in del_df.columns:
            ax.plot(del_df.index, del_df[agent_id], linestyle="--", linewidth=2)

    ax.set_title("Package Deliveries Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Deliveries")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return CompactFigureMatplotlib(fig)


@solara.component
def AnalyticsDashboard(model: BintWorldModel):
    with solara.lab.Tabs():
        with solara.lab.Tab("Verifiable Trust (TNFTs)"):
            TNFTsOverTime(model)
        with solara.lab.Tab("Total Deliveries"):
            DeliveriesOverTime(model)
        with solara.lab.Tab("Agent Points"):
            PointsOverTime(model)


bint = BintWorldModel()

page = SolaraViz(
    model=bint,
    # renderer=renderer,
    components=[DynamicMap, (AnalyticsDashboard, 1)],
    model_params=model_params,
    name="BINT Simulation",
)