import numpy as np
import mesa
from matplotlib.ticker import MaxNLocator
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle

from agents import DeliveryAgent, DropOffLocationAgent, MaliciousMapDeliveryAgent
from model import BintWorldModel
import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter


model_params = {
    # "num_agents": {
    #     "type": "SliderInt",
    #     "value": 5,
    #     "label": "Number of agents",
    #     "min": 1,
    #     "max": 20,
    #     "step": 1,
    # },
    "rng": {
        "type": "InputText",
        "value": 293276,
        "label": "Random seed",
    },
    "num_delivery": {
        "type": "SliderInt",
        "value": 8,
        "label": "Number of honest delivery agents",
        "min": -1,
        "max": 20,
        "step": 1,
    },
    "num_map_malicious": {
        "type": "SliderInt",
        "value": 2,
        "label": "Number of malicious agents",
        "min": -1,
        "max": 20,
        "step": 1,
    },
    "num_drop_offs": {
        "type": "SliderInt",
        "value": 3,
        "label": "Number of drop offs",
        "min": 1,
        "max": 20,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 50,
        "label": "Width",
        "min": 1,
        "max": 150,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": 50,
        "label": "Height",
        "min": 1,
        "max": 150,
        "step": 1,
    }
}


def agent_portrayal(agent: mesa.Agent):
    if isinstance(agent, MaliciousMapDeliveryAgent):
        return AgentPortrayalStyle(size=50, color="red")
    elif isinstance(agent, DeliveryAgent):
        return AgentPortrayalStyle(size=50, color="blue")
    elif isinstance(agent, DropOffLocationAgent):
        return AgentPortrayalStyle(size=50, marker="s", color="black")
    else:
        return None


def property_layer_portrayal(layer: mesa.discrete_space.PropertyLayer):
    if layer.name == "drop_off_locations":
        return PropertyLayerStyle(color="black", alpha=0.8)

    return None


bint = BintWorldModel(rng=293276)

renderer = SpaceRenderer(model=bint, backend="matplotlib")
renderer.draw_structure(lw=2, ls="solid", color="black", alpha=0.5)
renderer.setup_agents(agent_portrayal).draw_agents()


@solara.component
def ScoreBar(model: mesa.Model):
    update_counter.get()
    agent_ids = np.array(model.agents.select(agent_type=DeliveryAgent).get("unique_id"))
    agent_points = np.array(model.agents.select(agent_type=DeliveryAgent).get("points"))

    fig = Figure()
    ax = fig.subplots()
    ax.bar(agent_ids, agent_points, color="skyblue", edgecolor="black")
    ax.set_title("Scoreboard")
    ax.set_xlabel("Agent ID")
    ax.set_ylabel("Points")
    ax.set_ylim(bottom=0)

    return solara.FigureMatplotlib(fig)


@solara.component
def PointsOverTime(model: BintWorldModel):
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    df = model.datacollector.get_agenttype_vars_dataframe(DeliveryAgent)

    if not df.empty:
        # reshape to have steps on the x-axis, agent ids as different columns
        points_df = df.unstack(level="AgentID")["Points"]

        for agent_id in points_df.columns:
            ax.plot(points_df.index, points_df[agent_id], label=f"Agent {agent_id}")

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_title("Points Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Points")

    # Force axes to use integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return solara.FigureMatplotlib(fig)


page = SolaraViz(
    model=bint,
    renderer=renderer,
    components=[PointsOverTime],
    model_params=model_params,
    name="test"
)

page